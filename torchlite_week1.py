from __future__ import annotations
import os, sys
from typing import List
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# LangChain imports 
from langchain_community.document_loaders import NotionDBLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains import RetrievalQA
from langchain_core.messages import SystemMessage

# Load Notion database 
def load_notion_db() -> List:
    token = os.getenv("NOTION_API_KEY")
    db_id = os.getenv("NOTION_DATABASE_ID")
    if not (token and db_id):
        raise RuntimeError("Set NOTION_API_KEY and NOTION_DATABASE_ID in .env")

    loader = NotionDBLoader(
        integration_token=token,
        database_id=db_id,
        request_timeout_sec=30,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} page(s) from Notion DB {db_id[:8]}…")
    return docs

# Build vector store 
def build_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    supabase: Client = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_KEY"],
    )

    vectordb = SupabaseVectorStore.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        client=supabase,
        table_name="torchlite_embeddings",
        query_name="match_torchlite_embeddings",
    )
    print(f"Stored {len(chunks)} chunks in Supabase")
    return vectordb

# Ask a question
def answer_query(question: str, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 20})

    system = SystemMessage(
        content=(
            "You are AstroLabs' knowledge assistant. "
            "Answer the user's question using ONLY the provided sources. "
            "If the answer is not present, reply exactly: I don't know."
        )
    )

    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    return chain.invoke({"query": question})

# Main 
if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What is the focus of Week 1?"

    vectordb = build_vector_store(load_notion_db())
    result   = answer_query(query, vectordb)

    print("\n🟢 Answer:\n", result["result"], sep="")
    print("\n📄 Sources:")
    seen = set()
    for doc in result["source_documents"]:
        title = doc.page_content.splitlines()[0][:80]  # first heading / line
        if title not in seen:          # skip duplicates
            print("-", title)
            seen.add(title)
