from __future__ import annotations
import os, json, logging, hashlib, time, asyncio
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from notion_client import Client as NotionClient, APIResponseError
from langchain_community.document_loaders import NotionDBLoader
from loaders.notion_simple_page import NotionSimplePageLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from db_utils  import get_vectordb
from kb_version import bump_kb_version
import requests
import re, unicodedata

import collections, threading
_notn_lock   = threading.Lock()
_notn_calls  = collections.deque(maxlen=3)

# basic setup
load_dotenv()
logging.getLogger("langchain_community.document_loaders.notiondb").setLevel(logging.ERROR)
log = lambda m: print(f"[INGEST] {m}")

TOKENS        = [t.strip() for t in os.getenv("NOTION_API_KEYS",
                                              os.getenv("NOTION_API_KEY","")).split(",") if t.strip()]
CLIENTS       = [NotionClient(auth=t) for t in TOKENS]
QUERY         = os.getenv("NOTION_SEARCH_QUERY", "")
RESULT_LIMIT  = int(os.getenv("NOTION_RESULT_LIMIT", "0"))
FORCE_REBUILD = os.getenv("FORCE", "0") == "1"

STATE_FILE      = Path(".ingest_state.json")
MAX_CONCURRENCY = 3

# ── text normaliser ─────────────────────────────────────────
def _clean(txt: str) -> str:
    txt = unicodedata.normalize("NFKC", txt)        # tidy accents etc.
    txt = re.sub(r"[ \t]+\n", "\n", txt)            # trim line-end spaces
    txt = re.sub(r"\n{3,}", "\n\n", txt)            # max double newline
    txt = re.sub(r"[ \t]{2,}", " ", txt)            # collapse runs of spaces
    return txt.strip()

# ── usefulness heuristic ───────────────────────────────────
def _is_useful(txt: str, min_chars: int = 20, min_tokens: int = 5) -> bool:
    if len(txt.strip()) < min_chars:
        return False
    if not any(c.isalnum() for c in txt):           # all punctuation / emoji
        return False
    # (if you later add a tokenizer test, put it here)
    return True

# DEBUG ONLY
def debug_dump_blocks(block_id: str, client: NotionClient, depth: int = 0):
    """Recursively print block-id and type so we can see what the API returns."""
    indent = "│   " * depth
    try:
        resp = client.blocks.children.list(block_id=block_id, page_size=100)
    except Exception as e:
        print(indent + f"[ERROR] {block_id[:8]} {e}")
        return
    for blk in resp["results"]:
        t = blk["type"]
        print(indent + f"{blk['id'][:8]}  {t}")
        # Recurse if it claims to have children
        if blk.get("has_children"):
            debug_dump_blocks(blk["id"], client, depth + 1)

# state helpers
def load_state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}

def save_state(st: dict) -> None:
    STATE_FILE.write_text(json.dumps(st, indent=2))

def notion_guard():
    """Block only if the last 3 calls happened in <1 second."""
    with _notn_lock:
        now = time.time()
        if len(_notn_calls) == 3 and now - _notn_calls[0] < 1.0:
            sleep_for = 1.0 - (now - _notn_calls[0])
            time.sleep(sleep_for)
        _notn_calls.append(time.time())    

# Notion search (sync)
def search_one_ws(client: NotionClient, obj: str|None, limit: int) -> List[dict]:
    results, cursor = [], None
    while limit <= 0 or len(results) < limit:
        payload = {
            "query": QUERY,
            "sort": {"direction": "ascending", "timestamp": "last_edited_time"},
            "start_cursor": cursor,
            "page_size": 100,
        }
        if obj:
            payload["filter"] = {"value": obj, "property": "object"}
        try:
            notion_guard()
            resp = client.search(**payload)
        except Exception as e:
            if "start_cursor provided is invalid" in str(e):
                break
            log(f"  ⚠️  search error: {e}")
            break
        results.extend(resp["results"])
        if not resp.get("has_more"):
            break
        cursor = resp["next_cursor"]
    return results[:limit] if limit > 0 else results


# ─── Pull every row in a database safely & convert each row to docs ──────
def crawl_database_recursive(db_id: str, client: NotionClient) -> list[Document]:
    """
    Return Document objects for every row (page) in `db_id`, including:
      • A Markdown line for each property we care about
      • Any blocks that live inside the row page itself
    """
    docs, cursor = [], None
    while True:
        notion_guard()                                       # stay under 3 req/s
        resp = client.databases.query(
            db_id, page_size=100,
            **({"start_cursor": cursor} if cursor else {})
        )

        for row in resp["results"]:
            cell_lines: list[str] = []
            for name, val in row.get("properties", {}).items():
                t = val["type"]

                # ── basic text/number/url/checkbox ────────────────────────
                if t in ("title", "rich_text"):
                    txt = " ".join(r["plain_text"] for r in val[t]).strip()

                elif t == "number":
                    txt = str(val["number"]) if val["number"] is not None else ""

                elif t == "url":
                    txt = val["url"] or ""

                elif t == "checkbox":
                    txt = "Yes" if val["checkbox"] else "No"

                # ── NEW: metadata fields we now keep ─────────────────────
                elif t in ("select", "status"):
                    opt = val[t]            # may be None
                    txt = opt["name"] if opt and opt.get("name") else ""

                elif t == "multi_select":
                    options = val[t] or []  # None → []
                    txt = ", ".join(o.get("name", "") for o in options if o)

                elif t == "people":
                    users = val[t] or []    # None → []
                    txt = ", ".join(p.get("name", "") for p in users if p)

                elif t == "date":
                    d = val["date"]           # can be None
                    if d and d.get("start"):
                        txt = d["start"] + (f" → {d['end']}" if d.get("end") else "")
                    else:
                        txt = ""              # empty date → ignore


                else:                             # relation, roll-up, formula …
                    txt = ""                      # ignore for now

                if txt.strip():
                    cell_lines.append(f"**{name}**: {txt}")

            # save the row’s property summary iff it’s useful
            if cell_lines:
                docs.append(
                    Document(
                        page_content="\n".join(cell_lines),
                        metadata={"source_id": row["id"]},
                    )
                )

            # still crawl any blocks inside the row-page body
            docs += crawl_page_recursive(row["id"], client)

        if not resp.get("has_more"):
            break
        cursor = resp["next_cursor"]

    return docs

# ─────────────────────────────────────────────────────────────────────────

# ─── Recursive crawler that follows child pages & inline / linked DBs ───
def crawl_page_recursive(page_id: str, client: NotionClient) -> list[Document]:
    def block_to_markdown(blk: dict) -> str:
        t = blk["type"]
        plain = lambda key="rich_text": "".join(r["plain_text"] for r in blk[t].get(key, [])).strip()

        if t in ("heading_1", "heading_2", "heading_3"):
            level = {"heading_1": "#", "heading_2": "##", "heading_3": "###"}[t]
            return f"{level} {plain()}"
        if t == "paragraph":
            return plain()
        if t == "quote":
            return f"> {plain()}"
        if t in ("bulleted_list_item", "numbered_list_item", "to_do"):
            txt = plain()
            if not txt:
                return ""
            if t == "to_do":
                checked = "x" if blk[t].get("checked") else " "
                return f"- [{checked}] {txt}"
            prefix = "• " if t == "bulleted_list_item" else "1. "
            return f"{prefix}{txt}"
        if t in ("toggle", "synced_block"):
            return f"**{plain()}**"
        if t == "callout":
            return f"💡 {plain()}"
        if t == "table_row":
            def cells_md(cells): return " │ ".join("".join(r["plain_text"] for r in runs).strip() for runs in cells)
            return cells_md(blk["table_row"]["cells"])
        if t in ("table", "bookmark", "embed", "image", "video", "file", "pdf",
                 "link_preview", "divider", "template", "unsupported",
                 "table_of_contents", "equation"):
            return ""
        return ""

    # --------------- actual crawl -----------------
    docs, queue = [], [page_id]
    while queue:
        current = queue.pop()
        text_parts, cursor, db_ids = [], None, []
        while True:
            notion_guard()
            resp = client.blocks.children.list(block_id=current, page_size=100, start_cursor=cursor)
            for blk in resp["results"]:
                bt = blk["type"]
                if bt == "child_page":
                    queue.append(blk["id"])
                elif bt == "child_database":
                    db_ids.append(blk["id"])
                elif bt == "link_to_database":
                    db_ids.append(blk["link_to_database"]["database_id"])
                elif bt == "link_to_page":
                    tgt = blk["link_to_page"]
                    if "page_id" in tgt:
                        queue.append(tgt["page_id"])
                    elif "database_id" in tgt:
                        db_ids.append(tgt["database_id"])
                elif blk.get("has_children"):
                    queue.append(blk["id"])

                md = block_to_markdown(blk)
                if md:
                    text_parts.append(md)
            if not resp.get("has_more"):
                break
            cursor = resp["next_cursor"]

        for db_id in db_ids:
            docs += crawl_database_recursive(db_id, client)

        if text_parts:
            docs.append(Document("\n\n".join(text_parts), metadata={"source_id": current}))
    return docs

# ─────────────────────────────────────────────────────────────────────────


# single‐hit loader with one fallback
def _load_one(hit: dict, token: str) -> List[Document]:
    client = NotionClient(auth=token)

    def via_loader() -> List[Document]:
        if hit["object"] == "database":
            return crawl_database_recursive(hit["id"], client)
        else:
            return crawl_page_recursive(hit["id"], client)

    docs = via_loader()

    url = hit.get("url") or f"https://www.notion.so/{hit['id'].replace('-', '')}"
    for d in docs:
        d.metadata["source_url"] = url
        d.metadata["notion_id"] = hit["id"][:8]
    return docs


# async fan-out loader
ASYNC_TIMEOUT = 300

async def load_docs_async(hits: List[dict]) -> List[Document]:
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    results: List[List[Document]] = [None] * len(hits)

    async def worker(i: int, h: dict) -> None:
        async with sem:
            token = TOKENS[i % len(TOKENS)]
            try:
                docs = await asyncio.wait_for(
                    asyncio.to_thread(_load_one, h, token),
                    timeout=ASYNC_TIMEOUT
                )
            except asyncio.TimeoutError:
                log(f" [{i+1}/{len(hits)}] ⏱️  timeout on {h['id'][:8]}")
                docs = []
            except Exception as e:
                log(f" [{i+1}/{len(hits)}] ❌ error on {h['id'][:8]} → {e}")
                docs = []

            results[i] = docs
            url = h.get("url") or f"https://www.notion.so/{h['id'].replace('-', '')}"
            log(f" [{i+1}/{len(hits)}] ✓ {len(docs):3d} docs from {h['id'][:8]} → {url}")

    await asyncio.gather(*(worker(i, h) for i, h in enumerate(hits)))
    return [doc for sub in results if sub for doc in sub]


# ingestion driver
def gather_hits(prev: dict) -> List[dict]:
    log("Searching all Notion workspaces …")
    start = time.perf_counter()
    hits = []
    for c in CLIENTS:
        hits += search_one_ws(c, "database", RESULT_LIMIT)
        hits += search_one_ws(c, "page", RESULT_LIMIT)
    log(f"Found {len(hits)} objects in {time.perf_counter() - start:.2f}s")
    if FORCE_REBUILD:
        return hits
    return [h for h in hits if prev.get(h["id"]) != (h.get("last_edited_time") or "")]


def upsert(chunks: List[Document], batch=100):
    vdb = get_vectordb()
    total = len(chunks)
    log(f"Upserting {total} chunks …")
    for c in chunks:
        c.metadata = {k: v if isinstance(v, (str, int, float, bool)) or v is None else json.dumps(v, default=str)
                      for k, v in c.metadata.items()}
    for i in range(0, total, batch):
        end = min(i + batch, total)
        vdb.add_documents(chunks[i:end])
        pct = 100 * end / total
        try:
            count = vdb._collection.count()
            log(f"  ↑ {end}/{total} ({pct:.1f}%) vectors={count}")
        except AttributeError:
            log(f"  ↑ {end}/{total} ({pct:.1f}%) uploaded")
    log("Upsert done ✅")


def main():
    page_override = os.getenv("PAGE_ID")

    if page_override:
        log(f"Single-page ingest for {page_override[:8]} …")

        # (optional) keep this behind a flag if you ever need it again
        # if os.getenv("DEBUG_TREE_DUMP") == "1":
        #     debug_dump_blocks(page_override, NotionClient(auth=TOKENS[0]))
        #     print("──────────────────────────────────────────────────")

        docs = _load_one({"id": page_override, "object": "page"}, TOKENS[0])
        log(f"Found {len(docs)} doc(s) for page {page_override[:8]}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        log(f"Split into {len(chunks)} chunk(s)")

        upsert(chunks)
        log("✅ Single-page ingest done.")
        return

    state = load_state()
    hits = gather_hits(state)
    if not hits:
        return log("Nothing to do. ✅")

    log(f"Fetching {len(hits)} objects …")
    docs = asyncio.run(load_docs_async(hits))
    log(f"Loaded {len(docs)} docs")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    raw = splitter.split_documents(docs)
    seen, chunks = set(), []
    for c in raw:
        h = hashlib.sha256(c.page_content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            c.metadata["sha"] = h
            chunks.append(c)
    log(f"{len(chunks)} unique chunks")

    upsert(chunks)

    for h in hits:
        state[h["id"]] = h.get("last_edited_time") or ""
    save_state(state)

    log(f"KB version → {bump_kb_version()}")
    log("✅ Finished.")


if __name__ == "__main__":
    main()
