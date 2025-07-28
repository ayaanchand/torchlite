import os
from supabase import create_client, Client
import datetime as dt

def _client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL / SUPABASE_KEY in environment.")
    return create_client(url, key)

def get_kb_version(default: str = "v0") -> str:
    sb = _client()
    res = sb.table("config").select("value").eq("key", "kb_version").limit(1).execute()
    if res.data:
        return res.data[0]["value"]
    return default

def bump_kb_version() -> str:
    version = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    sb = _client()
    sb.table("config").upsert({"key": "kb_version", "value": version}).execute()
    return version
