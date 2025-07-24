from __future__ import annotations
import os, sys, logging, sys
from dotenv import load_dotenv
from supabase import create_client, Client

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import time

# Silence loader noise (shouldn't run anyway, but just in case)
logging.getLogger("langchain_community.document_loaders.notiondb").setLevel(logging.ERROR)

load_dotenv()

# Slow down answer generation
STREAM_DELAY = float(os.getenv("STREAM_DELAY", "0.03"))

MAX_CTX_CHARS   = int(os.getenv("MAX_CTX_CHARS", "6000"))

SUPABASE_URL   = os.environ["SUPABASE_URL"]
SUPABASE_KEY   = os.environ["SUPABASE_KEY"]
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "torchlite_embeddings")
TOP_K          = int(os.getenv("TOP_K", 3))

SYSTEM_TXT = (
    "You are AstroLabs' knowledge assistant. "
    "Answer the user's question using ONLY the provided sources. "
    "If the answer is not present, reply exactly: I don't know."
)

def get_vectordb() -> SupabaseVectorStore:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return SupabaseVectorStore(
        client=supabase,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        table_name=SUPABASE_TABLE,
        query_name=f"match_{SUPABASE_TABLE}",
    )

def answer_stream(question: str, vectordb):
    t0 = time.perf_counter()

    # --- super fast retrieval (single RPC) ---
    docs = vectordb.similarity_search(question, k=TOP_K)
    t1   = time.perf_counter()

    # --- cap context length to keep prompt small ---
    ctx_parts, running = [], 0
    for d in docs:
        txt_len = len(d.page_content)
        if running + txt_len > MAX_CTX_CHARS:
            break
        ctx_parts.append(d.page_content)
        running += txt_len
    context = "\n\n".join(ctx_parts)

    # --- prompt & chain ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TXT),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Context:\n{context}\n\n{user_question}"),
    ])
    chain = prompt | ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

    # --- stream answer ---
    print("🟢 Answer:\n", end="", flush=True)
    llm_start = time.perf_counter()
    for chunk in chain.stream({
        "messages":      [("human", question)],
        "context":       context,
        "user_question": question,
    }):
        txt = getattr(chunk, "content", "") or getattr(chunk, "delta", "")
        if txt:
            print(txt, end="", flush=True)
            if STREAM_DELAY > 0:
                time.sleep(STREAM_DELAY)
    llm_end = time.perf_counter()
    print("\n")

    # latency print
    total = llm_end - t0
    print(f"⏱️  retrieval: {t1 - t0:.2f}s | LLM: {llm_end - llm_start:.2f}s | total: {total:.2f}s\n")

    return docs

def main():
    question = " ".join(sys.argv[1:]) or "What is the focus of Week 1?"
    vectordb = get_vectordb()
    docs     = answer_stream(question, vectordb)

    print(f"📄 Top-{TOP_K} Source(s):")
    seen = set()
    for d in docs[:TOP_K]:
        title = d.page_content.splitlines()[0][:80]
        if title not in seen:
            print("-", title)
            seen.add(title)

if __name__ == "__main__":
    main()