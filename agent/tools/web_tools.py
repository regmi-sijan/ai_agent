import json
import urllib.request
import urllib.parse
import re
from typing import Optional, List, Dict, Any
from agent.tools.registry import register_tool
import config

@register_tool(
    name="simple_web_search",
    description="Simple, lightweight internet search engine to look up physics benchmarks, formulas, particle masses, or ML methods. Returns 2-3 concise snippets under 500 characters to prevent hitting usage or token limits."
)
def simple_web_search(query: str, max_results: int = 3) -> str:
    """Lightweight web search: uses DuckDuckGo with Wikipedia API fallback.
    Returns clean, concise snippets strictly limited in length to save tokens.
    """
    clean_query = query.strip()
    if not clean_query:
        return "Empty query provided."

    results = []

    # 1. Try DuckDuckGo first
    try:
        from duckduckgo_search import DDGS
        with DDGS(timeout=4) as ddgs:
            for r in ddgs.text(clean_query, max_results=max_results):
                title = r.get("title", "").strip()
                body = r.get("body", "").strip()
                if title and body:
                    results.append(f"{title}: {body[:140]}")
    except Exception:
        pass

    # 2. Fallback to Wikipedia Search API (100% free, reliable, no API key, fast)
    if not results:
        try:
            encoded_q = urllib.parse.quote(clean_query)
            wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_q}&format=json"
            req = urllib.request.Request(
                wiki_url,
                headers={"User-Agent": "PhysicsAgent/1.0 (lightweight search)"}
            )
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for item in data.get("query", {}).get("search", [])[:max_results]:
                    snippet = re.sub(r"<[^>]+>", "", item.get("snippet", "")).strip()
                    results.append(f"{item.get('title')}: {snippet[:140]}")
        except Exception:
            pass

    if not results:
        return f"No brief results found for: '{clean_query}'. Proceeding with standard analytical principles."

    # Return concise text capped to prevent context bloat
    output_text = "\n".join(results[:max_results])
    return output_text[:600]


# Backward-compatible lightweight wrappers for old tool names
@register_tool(
    name="search_physics_literature",
    description="Search the web for standard formulas, kinematic relations, or particle properties (lightweight)."
)
def search_physics_literature(query: str, max_results: int = 3) -> str:
    return simple_web_search(query, max_results=max_results)


@register_tool(
    name="lookup_physics_benchmarks",
    description="Look up particle masses, constants, or resonance properties (lightweight)."
)
def lookup_physics_benchmarks(topic_or_entity: str) -> str:
    return simple_web_search(topic_or_entity, max_results=2)


@register_tool(
    name="query_arxiv_physics_papers",
    description="Search for physics preprints and formulas (lightweight)."
)
def query_arxiv_physics_papers(query: str, max_results: int = 2) -> str:
    return simple_web_search(query, max_results=max_results)


@register_tool(
    name="fetch_physics_webpage",
    description="Fetch a short excerpt from a URL (lightweight)."
)
def fetch_physics_webpage(url: str) -> str:
    return f"URL preview: {url} (Webpage fetching disabled to prevent token usage limit; use simple_web_search instead)."
