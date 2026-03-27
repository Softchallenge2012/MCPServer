from fastmcp import FastMCP
from web_scrape_tool import web_scrape_function
from web_search_tool import web_search_function

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

@mcp.tool()
def web_scrape(url: str) -> dict:
    """Scrape a website"""
    return web_scrape_function(url)

@mcp.tool()
def web_search(query: str) -> dict:
    """Search the web"""
    return web_search_function(query)

if __name__ == "__main__":
    mcp.run(transport="stdio")
    # mcp.run(transport="http", host="localhost", port=8000)