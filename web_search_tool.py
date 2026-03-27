"""
Web Search Tool - Search the web using multiple providers.

Supports:
- Google Custom Search API (GOOGLE_API_KEY + GOOGLE_CSE_ID)
- Brave Search API (BRAVE_SEARCH_API_KEY)

Auto-detection: If provider="auto", tries Brave first (backward compatible), then Google.
"""

from __future__ import annotations

import os
import time
from typing import Literal

import httpx

def _search_google(
    query: str,
    num_results: int,
    country: str,
    language: str,
    api_key: str,
    cse_id: str,
) -> dict:
    """Execute search using Google Custom Search API."""
    max_retries = 3
    for attempt in range(max_retries + 1):
        response = httpx.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": api_key,
                "cx": cse_id,
                "q": query,
                "num": min(num_results, 10),
                "lr": f"lang_{language}",
                "gl": country,
            },
            timeout=30.0,
        )

        if response.status_code == 429 and attempt < max_retries:
            time.sleep(2**attempt)
            continue

        if response.status_code == 401:
            return {"error": "Invalid Google API key"}
        elif response.status_code == 403:
            return {"error": "Google API key not authorized or quota exceeded"}
        elif response.status_code == 429:
            return {"error": "Google rate limit exceeded. Try again later."}
        elif response.status_code != 200:
            return {"error": f"Google API request failed: HTTP {response.status_code}"}

        break

    data = response.json()
    results = []
    for item in data.get("items", [])[:num_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
        )

    return {
        "query": query,
        "results": results,
        "total": len(results),
        "provider": "google",
    }

def _search_brave(
    query: str,
    num_results: int,
    country: str,
    api_key: str,
) -> dict:
    """Execute search using Brave Search API."""
    max_retries = 3
    for attempt in range(max_retries + 1):
        response = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={
                "q": query,
                "count": min(num_results, 20),
                "country": country,
            },
            headers={
                "X-Subscription-Token": api_key,
                "Accept": "application/json",
            },
            timeout=30.0,
        )

        if response.status_code == 429 and attempt < max_retries:
            time.sleep(2**attempt)
            continue

        if response.status_code == 401:
            return {"error": "Invalid Brave API key"}
        elif response.status_code == 429:
            return {"error": "Brave rate limit exceeded. Try again later."}
        elif response.status_code != 200:
            return {"error": f"Brave API request failed: HTTP {response.status_code}"}

        break

    data = response.json()
    results = []
    for item in data.get("web", {}).get("results", [])[:num_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
            }
        )

    return {
        "query": query,
        "results": results,
        "total": len(results),
        "provider": "brave",
    }

def _get_credentials() -> dict:
    """Get available search credentials."""
    return {
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "google_cse_id": os.getenv("GOOGLE_CSE_ID"),
        "brave_api_key": os.getenv("BRAVE_SEARCH_API_KEY"),
    }

def web_search_function(
    query: str,
    num_results: int = 10,
    country: str = "us",
    language: str = "en",
    provider: Literal["auto", "google", "brave"] = "auto",
    env_var: dict | None = None,
) -> dict:
    if not query or len(query) > 500:
        return {"error": "Query must be 1-500 characters"}

    # Determine credentials source: passed env dict vs server environment
    source = env_var if env_var is not None else _get_credentials()
    
    final_google_api_key = source.get("GOOGLE_API_KEY")
    final_google_cse_id = source.get("GOOGLE_CSE_ID")
    final_brave_api_key = source.get("BRAVE_SEARCH_API_KEY")

    google_available = final_google_api_key and final_google_cse_id
    brave_available = bool(final_brave_api_key)

    try:
        if provider == "google":
            if not google_available:
                return {"error": "Google credentials not provided in env_var or configuration"}
            return _search_google(query, num_results, country, language, final_google_api_key, final_google_cse_id)
        elif provider == "brave":
            if not brave_available:
                return {"error": "Brave credentials not provided in env_var or configuration"}
            return _search_brave(query, num_results, country, final_brave_api_key)
        else: # auto
            if brave_available:
                return _search_brave(query, num_results, country, final_brave_api_key)
            elif google_available:
                return _search_google(query, num_results, country, language, final_google_api_key, final_google_cse_id)
            else:
                return {"error": "No search credentials provided in env_var or configuration"}

    except httpx.TimeoutException:
        return {"error": "Search request timed out"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {str(e)}"}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}

if __name__ == "__main__":
    results = web_search('bacterial vaginosis')
    print(results)
