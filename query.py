from __future__ import annotations
import os, sys, time, logging, random
from dotenv import load_dotenv


from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langfuse.langchain import CallbackHandler
from typing import List, Literal, Optional
from db_utils import get_vectordb
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from kb_version import get_kb_version  # reads kb_version from Supabase

logging.getLogger("langchain_community.document_loaders.notiondb").setLevel(logging.ERROR)

load_dotenv()

# Toggles & config 
STREAM_DELAY   = float(os.getenv("STREAM_DELAY", "0.03"))
MAX_CTX_CHARS  = int(os.getenv("MAX_CTX_CHARS", "12000"))
TOP_K          = int(os.getenv("TOP_K", "4"))
MAX_HISTORY_TURNS= int(os.getenv("MAX_HISTORY_TURNS", "8"))  # keep last N turns

SUPABASE_URL   = os.environ["SUPABASE_URL"]
SUPABASE_KEY   = os.environ["SUPABASE_KEY"]
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "torchlite_embeddings")

# Tracing switches
TRACE_ENABLED  = os.getenv("TRACE_ENABLED", "1") == "1"
SAMPLE_RATE    = float(os.getenv("LANGFUSE_SAMPLING_RATE", "1"))

# Models 
LLM_MODEL      = os.getenv("LLM_MODEL", "gpt-4o")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")

ASTROLABS_BLURB = (
    "You are chatting with AstroLabs employees. "
    "AstroLabs is a company that provides support for businesses expanding into Saudi Arabia and the broader MENA region. "
    "They offer services like business setup, licensing, and guidance on market entry and expansion. "
    "AstroLabs also focuses on supporting startups and driving growth through various programs and partnerships.\n"
)

SYSTEM_TXT = (
    "You are AstroLabs’ internal knowledge assistant. Your audience is AstroLabs employees.\n"
    "\n"
    "Voice & format:\n"
    "- Write in clear, continuous prose (short paragraphs). Do not use bullets, numbering, or section headings.\n"
    "- Do not mention or allude to where information came from (no 'context', 'background', 'documents', or 'sources').\n"
    "\n"
    "Grounding & safety:\n"
    "- Use only the material you are given as evidence. If a specific item (number, URL, name, date, price, or policy detail) is not present, say so plainly.\n"
    "- Never invent URLs, website links, names, headcounts, prices, or policy details. Include a URL only if it appears verbatim in the material.\n"
    "- Do not output secrets, credentials, or personal data. If a request seems restricted, remind the user to use proper access channels.\n"
    "\n"
    "Question type (decide silently):\n"
    "- FACTUAL: asks for a specific data point or fact (e.g., counts, dates, names, policy clauses, URLs).\n"
    "- GUIDANCE: asks for advice, judgment, process, or next steps (e.g., 'how should we…', 'what’s the best way…', 'help me…').\n"
    "\n"
    "Decision rule:\n"
    "- If the question is FACTUAL and the material clearly contains the answer, provide a detailed answer in continuous prose.\n"
    "- If the question is FACTUAL and the material is missing or ambiguous, ask exactly ONE concise clarifying question and STOP. Do not provide general guidance.\n"
    "- If the question is GUIDANCE, you may provide internal best-practice guidance. If anything is unclear, ask ONE concise clarifying question first,\n"
    "  then give a detailed guidance answer in prose, stating assumptions and where to verify internally. Do not invent AstroLabs-specific facts.\n"
)

def _to_langchain_history(history: Optional[List[dict]]) -> List:
    if not history:
        return []
    out: List = []
    # keep only last N turns to control tokens
    trimmed = history[-MAX_HISTORY_TURNS:] if len(history) > MAX_HISTORY_TURNS else history
    for m in trimmed:
        r = (m.get("role") or "").strip()
        c = (m.get("content") or "").strip()
        if not c:
            continue
        if r == "system":
            out.append(SystemMessage(content=c))
        elif r == "user":
            out.append(HumanMessage(content=c))
        elif r == "assistant":
            out.append(AIMessage(content=c))
    return out

# Langfuse callback
langfuse_handler = CallbackHandler()

def get_callbacks():
    if not TRACE_ENABLED:
        return []
    return [langfuse_handler] if random.random() < SAMPLE_RATE else []

# Vector DB 
vectordb = get_vectordb()

# Answer (streaming)
def answer_stream(question: str, vectordb, chat_history: Optional[List[dict]] = None):
    t0 = time.perf_counter()

    # Retrieval
    docs = vectordb.similarity_search(question, k=TOP_K)
    t1   = time.perf_counter()

    # Build bounded context
    ctx_parts, running = [], 0
    for d in docs:
        txt = d.page_content
        if running + len(txt) > MAX_CTX_CHARS:
            break
        ctx_parts.append(txt)
        running += len(txt)
    context = "\n\n".join(ctx_parts)

    # Prompt + LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TXT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human",
        "Use the BACKGROUND silently. Do not mention that you used any background or documents.\n\n"
        "BACKGROUND:\n{context}\n\n"
        "QUESTION (from an AstroLabs employee): {user_question}\n\n"
        "Requirements:\n"
        "1) If the background clearly answers a FACTUAL question, respond with a detailed answer in continuous prose (no bullets/headings).\n"
        "2) If the question is FACTUAL but the background is insufficient or ambiguous, ask ONE concise clarifying question and stop.\n"
        "3) If the question is GUIDANCE, optionally ask ONE concise clarifying question if needed, then provide a detailed best-practice answer in continuous prose,\n"
        "   stating assumptions and how/where to verify internally. Do not include any URL or specific figure that does not appear verbatim in the background.\n"
        ),
    ])

    chain = prompt | ChatOpenAI(model_name=LLM_MODEL, temperature=0, max_tokens=1600, streaming=True)

    callbacks   = get_callbacks()
    kb_version  = get_kb_version()

    # Stream answer
    # print("🟢 Answer:\n", end="", flush=True)
    llm_start = time.perf_counter()
    answer_parts = []

    for chunk in chain.stream(
        {
            "context":       context,
            "user_question": question,
            "chat_history":  _to_langchain_history(chat_history),
        },
        config={
            "callbacks": get_callbacks(),
            "metadata": {
                "app": "torchlite",
                "env": os.getenv("APP_ENV", "dev"),
                "table": SUPABASE_TABLE,
                "kb_version": get_kb_version(),
                "model": LLM_MODEL,
                "embed_model": EMBED_MODEL,
            },
            "tags": ["torchlite", "retrieval-qa"],
        },
    ):
        token = getattr(chunk, "content", "") or getattr(chunk, "delta", "") or ""
        if not token:
            continue
        answer_parts.append(token)
        print(token, end="", flush=True)
        if STREAM_DELAY > 0:
            time.sleep(STREAM_DELAY)

    print()  # final newline
    llm_end = time.perf_counter()
    answer = "".join(answer_parts)
    # print("\n")

    # Latency
    # total = llm_end - t0
    # print(f"⏱️  retrieval: {t1 - t0:.2f}s | LLM: {llm_end - llm_start:.2f}s | total: {total:.2f}s\n")

    return answer, docs

# Main 
def main():
    question = " ".join(sys.argv[1:]) or "What is the focus of Week 1?"
    ans, docs = answer_stream(question, vectordb, chat_history=[])

    print(f"📄 Top-{TOP_K} source(s):")

    seen = set()
    for d in docs[:TOP_K]:
        url   = d.metadata.get("source_url", "URL-missing")
        title = d.page_content.splitlines()[0][:80]
        if url not in seen:                 # avoid printing duplicates
            print(f"- {title}  →  {url}")
            seen.add(url)

if __name__ == "__main__":
    main()
