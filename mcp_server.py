from fastmcp import FastMCP
from typing import Iterable, List, Optional

from web_scrape_tool import web_scrape_function
from web_search_tool import web_search_function
from gpt_tool import chat
from db_tool import QdrantDB, ChromaDB


mcp = FastMCP("MEDQA_MCP")

# ---------------------------------------------------------------------------
# Module-level registry: DB instances keyed by a caller-chosen string name.
# MCP tools can only exchange JSON-serializable values, so objects can never
# be tool parameters.  Instead every tool takes a `db_name` string and looks
# up (or stores) the live instance here.
# ---------------------------------------------------------------------------
_qdrant_registry: dict[str, QdrantDB] = {}
_chroma_registry:  dict[str, ChromaDB]  = {}


# ── Qdrant tools ────────────────────────────────────────────────────────────

@mcp.tool()
def create_qdrant_db(
    db_name: str,
    collection_name: str,
    model_name: str = "vertex-embedding",
    host: Optional[str] = None,
    port: Optional[int] = None,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    encoding_name: str = "cl100k_base",
    context_length: int = 1000,
) -> str:
    """Create (or re-create) a Qdrant collection and register it under *db_name*."""
    qdb = QdrantDB(
        model_name,
        host=host,
        port=port,
        url=url,
        api_key=api_key,
        encoding_name=encoding_name,
        context_length=context_length,
    )
    try:
        qdb.delete_collection(collection_name=collection_name)
    except Exception:
        pass
    qdb.create_collection(collection_name=collection_name)
    _qdrant_registry[db_name] = qdb
    return f"Qdrant DB '{db_name}' created with collection '{collection_name}'."


@mcp.tool()
def add_texts_qdrant(
    db_name: str,
    texts: List[str],
    metadatas: Optional[List[str]] = None,
) -> List[str]:
    """Add texts to the named Qdrant database."""
    qdb = _qdrant_registry.get(db_name)
    if qdb is None:
        raise ValueError(f"Qdrant DB '{db_name}' not found. Call create_qdrant_db first.")
    return qdb.add_texts(texts, metadatas)


@mcp.tool()
def search_qdrant(
    db_name: str,
    text: str = "",
    top_k: int = 5,
) -> dict:
    """Search the named Qdrant database."""
    qdb = _qdrant_registry.get(db_name)
    if qdb is None:
        raise ValueError(f"Qdrant DB '{db_name}' not found. Call create_qdrant_db first.")
    return qdb.search(text, top_k=top_k)


# ── Chroma tools ─────────────────────────────────────────────────────────────

@mcp.tool()
def create_chroma_db(
    db_name: str,
    db_dir: str = "data/chromadb",
    model_name: str = "gpt-4o",
    embed_name: str = "vertex-embedding",
) -> str:
    """Create (or re-create) a Chroma database and register it under *db_name*."""
    cdb = ChromaDB(db_dir=db_dir, model_name=model_name, embed_name=embed_name)
    cdb.create_db(db_dir)
    _chroma_registry[db_name] = cdb
    return f"Chroma DB '{db_name}' created at '{db_dir}'."


@mcp.tool()
def add_texts_chroma(
    db_name: str,
    texts: List[str],
    metadatas: Optional[List[str]] = None,
) -> List[str]:
    """Add texts to the named Chroma database."""
    cdb = _chroma_registry.get(db_name)
    if cdb is None:
        raise ValueError(f"Chroma DB '{db_name}' not found. Call create_chroma_db first.")
    return cdb.add_texts(texts, metadatas)


@mcp.tool()
def search_chroma(
    db_name: str,
    text: str = "",
) -> dict:
    """Search the named Chroma database."""
    cdb = _chroma_registry.get(db_name)
    if cdb is None:
        raise ValueError(f"Chroma DB '{db_name}' not found. Call create_chroma_db first.")
    return cdb.search(text)


# ── Other tools ──────────────────────────────────────────────────────────────

@mcp.tool()
def gpt_chat(query: str, model: str, env_var: dict | None = None) -> str:
    """Chat with GPT"""
    return chat(query, model, env_var)

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

@mcp.tool()
def web_scrape(url: str, env_var: dict | None = None) -> dict:
    """Scrape a website"""
    return web_scrape_function(url, env_var=env_var)

@mcp.tool()
def web_search(query: str, env_var: dict | None = None) -> dict:
    """Search the web"""
    return web_search_function(query, env_var=env_var)


if __name__ == "__main__":
    mcp.run(transport="stdio")
    # mcp.run(transport="http", host="localhost", port=8000)