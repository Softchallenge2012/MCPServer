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