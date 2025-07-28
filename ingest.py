from __future__ import annotations
import os, json, logging, hashlib, time
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from notion_client import Client as NotionClient
from supabase import create_client, Client

from langchain_community.document_loaders import NotionDBLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from loaders.notion_simple_page import NotionSimplePageLoader
from kb_version import bump_kb_version   # ← NEW

# quiet warnings
logging.getLogger("langchain_community.document_loaders.notiondb").setLevel(logging.ERROR)

load_dotenv()

NOTION_API_KEY      = os.environ["NOTION_API_KEY"]
SUPABASE_URL        = os.environ["SUPABASE_URL"]
SUPABASE_KEY        = os.environ["SUPABASE_KEY"]
SUPABASE_TABLE      = os.getenv("SUPABASE_TABLE", "torchlite_embeddings")
NOTION_SEARCH_QUERY = os.getenv("NOTION_SEARCH_QUERY", "")
RESULT_LIMIT        = int(os.getenv("NOTION_RESULT_LIMIT", 50))
FORCE_REBUILD       = os.getenv("FORCE", "0") == "1"

STATE_FILE = Path(".ingest_state.json")
notion = NotionClient(auth=NOTION_API_KEY)

def log(msg: str):
    print(f"[INGEST] {msg}")

# state helpers 
def load_state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}

def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))

# Notion 
def search_notion(query: str, obj: str | None, limit: int) -> List[dict]:
    results, cursor = [], None
    while len(results) < limit:
        payload = {
            "query": query,
            "sort": {"direction": "ascending", "timestamp": "last_edited_time"},
            "start_cursor": cursor,
        }
        if obj:
            payload["filter"] = {"value": obj, "property": "object"}
        resp = notion.search(**payload)
        results.extend(resp["results"])
        if not resp.get("has_more"):
            break
        cursor = resp["next_cursor"]
    return results[:limit]

def doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def gather_changed_hits(state: dict) -> List[dict]:
    log("Searching Notion …")
    t0 = time.perf_counter()
    hits = search_notion(NOTION_SEARCH_QUERY, "database", RESULT_LIMIT) + \
           search_notion(NOTION_SEARCH_QUERY, "page",     RESULT_LIMIT)
    log(f"Found {len(hits)} objects in {time.perf_counter()-t0:.2f}s")

    if FORCE_REBUILD:
        log("FORCE rebuild enabled – processing all objects.")
        return hits

    changed = []
    for h in hits:
        last_edit = h.get("last_edited_time") or ""
        if state.get(h["id"]) != last_edit:
            changed.append(h)

    log(f"{len(changed)} object(s) changed since last ingest.")
    return changed

def load_docs_for_hits(hits: List[dict]) -> List[Document]:
    docs: List[Document] = []
    for i, h in enumerate(hits, 1):
        obj_type = h["object"]
        short_id = h["id"][:8]
        log(f" [{i}/{len(hits)}] Loading {obj_type} {short_id} …")
        try:
            if obj_type == "database":
                loader = NotionDBLoader(
                    integration_token=NOTION_API_KEY,
                    database_id=h["id"],
                    request_timeout_sec=30,
                )
            else:
                loader = NotionSimplePageLoader(
                    integration_token=NOTION_API_KEY,
                    page_id=h["id"],
                    timeout=30,
                )
            loaded = loader.load()
            log(f"   ✓ {len(loaded)} doc(s)")
            docs.extend(loaded)
        except Exception as e:
            log(f"   ⚠️  skipped ({e})")
    return docs

def upsert_chunks(chunks: List[Document]):
    log("Connecting to Supabase …")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name=SUPABASE_TABLE,
        query_name=f"match_{SUPABASE_TABLE}",
    )

    log(f"Upserting {len(chunks)} chunk(s) …")
    t0 = time.perf_counter()
    vectordb.add_documents(chunks)
    log(f"Upsert done in {time.perf_counter()-t0:.2f}s")

def main():
    t_start = time.perf_counter()
    state = load_state()

    hits_to_process = gather_changed_hits(state)
    if not hits_to_process:
        log("Nothing changed. ✅ Done.")
        return

    t0 = time.perf_counter()
    docs = load_docs_for_hits(hits_to_process)
    log(f"Loaded {len(docs)} docs in {time.perf_counter()-t0:.2f}s")

    if not docs:
        log("No docs loaded (permissions?). Exiting.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=0)
    t0 = time.perf_counter()
    raw_chunks = splitter.split_documents(docs)
    log(f"Chunked into {len(raw_chunks)} pieces in {time.perf_counter()-t0:.2f}s")

    # de-dup within batch
    seen, unique_chunks = set(), []
    for c in raw_chunks:
        h = doc_hash(c.page_content)
        if h in seen:
            continue
        seen.add(h)
        c.metadata = {**c.metadata, "sha": h}
        unique_chunks.append(c)
    if len(unique_chunks) != len(raw_chunks):
        log(f"Removed {len(raw_chunks) - len(unique_chunks)} duplicate chunk(s).")

    upsert_chunks(unique_chunks)

    # Update state
    for h in hits_to_process:
        state[h["id"]] = h.get("last_edited_time") or ""
    save_state(state)

    # Bump KB version so caches are invalidated automatically  ← NEW
    new_ver = bump_kb_version()
    log(f"KB version bumped → {new_ver}")

    log(f"✅ Finished. Total time {time.perf_counter()-t_start:.2f}s.")

if __name__ == "__main__":
    main()
