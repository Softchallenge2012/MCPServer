from fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from web_scrape_tool import web_scrape_function
from web_search_tool import web_search_function
from gpt_tool import chat
from db_tool import *


mcp = FastMCP("MEDQA_MCP")

@mcp.tool()
def create_qdrant_db(collection_name, model_name = 'vertex-embedding', host=None, port=None, url=None, api_key=None, encoding_name="cl100k_base", context_length=1000) -> None:
    """Create a Qdrant database"""

    qdrant_db = QdrantDB(model_name, host=host, port=port, url=url, api_key=api_key, encoding_name=encoding_name, context_length=context_length)
    # 1. Delete the collection (handles error if it doesn't exist)
    qdrant_db.delete_collection(collection_name=collection_name)

    # 2. Create the collection from scratch
    qdrant_db.create_collection(collection_name=collection_name)

    return qdrant_db

@mcp.tool()
def add_texts_qdrant(qdrant_db: QdrantClient, texts: Iterable[str], metadatas: Optional[List[str]] = None) -> List[str]:
    """Add texts to the Qdrant database"""
    if qdrant_db is None:
        raise ValueError("Qdrant database not found. Please create it first.")
    return qdrant_db.add_texts(texts, metadatas)

@mcp.tool()
def search_qdrant(qdrant_db: QdrantClient, text: str = "", top_k: int = 5, model=None):
    """Search the Qdrant database"""
    if qdrant_db is None:
        raise ValueError("Qdrant database not found. Please create it first.")
    return qdrant_db.search(text, top_k, model)

@mcp.tool()
def create_chroma_db(db_dir: str = 'data/chromadb', model_name: str = 'gpt-4o', embed_name: str = 'vertex-embedding') -> None:
    """Create a Chroma database"""
    llm = ChatOpenAI(
        model="gpt-4o",                # The model identifier set in your LiteLLM config
        api_key="sk-1234",                    # Your LiteLLM virtual key
        base_url="http://0.0.0.0:4000",       # The LiteLLM Proxy URL
        temperature=0
    )
    cdb = ChromaDB()

    cdb.create_db(directory_cdb)
    return cdb

@mcp.tool()
def add_texts_chroma(chroma_db: Chroma, texts: Iterable[str], metadatas: Optional[List[str]] = None) -> List[str]:
    """Add texts to the Chroma database"""
    return chroma_db.add_texts(texts, metadatas)

@mcp.tool()
def search_chroma(chroma_db: Chroma, text: str = "", model=None):
    """Search the Chroma database"""
    return chroma_db.search(text, model)

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
def web_search(
    query: str, 
    env_var: dict | None = None
) -> dict:
    """Search the web"""
    return web_search_function(
        query, 
        env_var=env_var
    )

if __name__ == "__main__":
    mcp.run(transport="stdio")
    # mcp.run(transport="http", host="localhost", port=8000)