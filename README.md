# Torchlite (Backend)

Internal knowledge assistant for AstroLabs.  
Ingests Notion → creates embeddings in Supabase (pgvector) → retrieves top snippets via RPC → streams grounded answers.

---

## 1) Features

- **Ingestion** from Notion (pages & DB rows), normalization, chunking, dedup by **SHA-256**  
- **Embeddings** with OpenAI `text-embedding-3-small` (configurable)  
- **Vector search** via Supabase Postgres + pgvector (HNSW)  
- **Two RPCs** for retrieval: with and without metadata filters  
- **Answering** with OpenAI LLMs (configurable) in **continuous prose**  
- **Versioning** with `kb_version` to tag deployments/ingests  
- Optional: **Langfuse** traces, **Sentry** errors

---

## 2) Repo structure

```
torchlite/
  api.py                 # FastAPI app (HTTP API)
  ingest.py              # Notion → chunks → embeddings → Supabase
  query.py               # CLI: ask a question via retrieval + LLM
  db_utils.py            # Vector backend (Supabase or Chroma dev)
  diagnose_rpc.py        # Quick check for RPC/role/timeouts
  kb_version.py          # Read/bump KB version in DB
  loaders/
    notion_simple_page.py
  chroma_db/             # (dev only, if VECTOR_BACKEND=chroma)
  .env                   # local config (not committed)
  requirements.txt
  litellm.yaml           # (optional) model routing
```

---

## 3) Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # or create .env with the vars below
```

Run the API:

```bash
uvicorn api:app --reload --port 8000
```

Test retrieval + answer (CLI):

```bash
python query.py "How do we onboard a new member?"
```

Ingest a single Notion page:

```bash
PAGE_ID="<notion-page-id>" python ingest.py
```

Full crawl:

```bash
python ingest.py
```

---

## 4) Configuration (.env)

```ini
# Core DB
SUPABASE_URL=...
SUPABASE_KEY=...                 
SUPABASE_TABLE=torchlite_embeddings

# Models
OPENAI_API_KEY=...
LLM_MODEL=gpt-4o
EMBED_MODEL=text-embedding-3-small

# Retrieval
VECTOR_BACKEND=supabase           # or chroma (dev)
TOP_K=4
MAX_CTX_CHARS=12000
STREAM_DELAY=0.03
MAX_HISTORY_TURNS=8

# Notion
NOTION_API_KEYS=...               # comma-separated if multiple
NOTION_SEARCH_QUERY=
NOTION_RESULT_LIMIT=0
FORCE=0                           # set 1 to force rebuild
PAGE_ID=                          # set for single-page ingest

# Observability (optional)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=
SENTRY_DSN=
APP_ENV=dev
TRACE_ENABLED=1
```
---

## 5) Supabase / Postgres setup

### Table

```sql
create table if not exists public.torchlite_embeddings (
  id         uuid primary key default gen_random_uuid(),
  content    text not null,
  metadata   jsonb not null default '{}'::jsonb,
  embedding  vector(1536) not null,
  created_at timestamptz not null default now()
);
```

### Index (HNSW, cosine)

```sql
create index if not exists idx_torchlite_embedding_hnsw
on public.torchlite_embeddings
using hnsw (embedding vector_cosine_ops)
with (m = 8, ef_construction = 32);

create index if not exists idx_torchlite_metadata_gin
on public.torchlite_embeddings using gin (metadata jsonb_path_ops);

analyze public.torchlite_embeddings;
```

### RPCs (both signatures)

```sql
-- 2-arg
create or replace function public.match_torchlite_embeddings(
  query_embedding vector,
  match_count int default 20
) returns table (id uuid, content text, metadata jsonb, similarity float)
language sql stable
set statement_timeout='30s'
set lock_timeout='0'
set idle_in_transaction_session_timeout='0'
as $$
  select e.id, e.content, e.metadata,
         1 - (e.embedding <=> query_embedding) as similarity
  from public.torchlite_embeddings e
  order by e.embedding <-> query_embedding
  limit greatest(1, match_count);
$$;

-- 3-arg (filtered)
drop function if exists public.match_torchlite_embeddings(vector, integer, jsonb);
create function public.match_torchlite_embeddings(
  query_embedding vector,
  match_count int,
  filter jsonb
) returns table (id uuid, content text, metadata jsonb, similarity float)
language sql stable
set statement_timeout='30s'
set lock_timeout='0'
set idle_in_transaction_session_timeout='0'
as $$
  select e.id, e.content, e.metadata,
         1 - (e.embedding <=> query_embedding) as similarity
  from public.torchlite_embeddings e
  where (filter is null) or (e.metadata @> filter)
  order by e.embedding <-> query_embedding
  limit greatest(1, match_count);
$$;

---

## 6) Ingestion

- `ingest.py` searches Notion (pages + DBs), crawls children, normalizes to markdown, splits (`chunk_size=1000`, `overlap=150`), dedups by **SHA-256**, embeds, and upserts to Supabase.
- Use **`NOTION_API_KEYS`** (comma separated) to increase throughput with light rate limiting.
- After large upserts, Postgres `ANALYZE` runs to keep the planner healthy.  
- `kb_version` is bumped at the end of a successful run.

---

## 7) Answering

- `query.py` builds context from top-K retrieved chunks (bounded by `MAX_CTX_CHARS`) and prompts the LLM to answer in **continuous prose**.  
- Factual questions require grounding in the retrieved context; if missing/ambiguous, it asks **one** clarifying question.  
- Guidance questions are allowed (best practices) but **no invented AstroLabs-specific facts**.

---

## 8) API (FastAPI)

Common pattern (from `api.py`):

```
GET /healthz          → 200 OK
POST /answer {q}      → streams answer tokens
```

- Retrieval uses `SupabaseVectorStore.similarity_search(question, k=TOP_K)` which calls the RPCs above.
- Responses include minimal metadata (e.g., `kb_version`) for tracing.

---

## 9) Troubleshooting

**Instant timeout / 57014 (“canceling statement due to statement timeout”)**  
- Ensure function-level `SET statement_timeout='30s'` is present in both RPCs.  
- Confirm `diagnose_rpc.py` returns your role/timeout (anon often shows `30s`).  
- Check that the **HNSW index** exists; without it, large scans may exceed timeouts.

**“function … (vector, integer) is not unique” / defaults issue**  
- Drop the old 3-arg function before recreating (Postgres can’t remove defaults from an existing signature):  
  `drop function public.match_torchlite_embeddings(vector, integer, jsonb);`

**HNSW build memory error (maintenance_work_mem)**  
- Lower index params: `m=8`, `ef_construction=32` (used above).  
- Recreate the index and `ANALYZE`.

**Slow retrieval**  
- Verify `ORDER BY embedding <-> query_embedding` (uses index).  
- Ensure `vector_cosine_ops` matches your similarity math.  
- `ANALYZE` after big ingests.

---

## 10) Security

- **Server-side only**: Supabase/OpenAI keys are used in the backend; the frontend never sees secrets.  
- **Least privilege**: grant `EXECUTE` on RPCs; avoid raw `SELECT` on the table for public roles.  

---

## 11) License / ownership

Internal AstroLabs project. Do not distribute without permission.

---

### Appendix: smoke tests

**SQL (console):**

```sql
-- 2-arg
select * from public.match_torchlite_embeddings(
  (select embedding from public.torchlite_embeddings limit 1), 5
);

-- 3-arg
select * from public.match_torchlite_embeddings(
  (select embedding from public.torchlite_embeddings limit 1), 5, null::jsonb
);
```

**CLI:**
```bash
python query.py "For a Saudi LLC we helped incorporate, what are the ZATCA zakat requirements?"
```

