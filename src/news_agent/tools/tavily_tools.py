import os
from typing import Any, Dict, List, Literal

from langchain_core.tools import tool
from tavily import TavilyClient


@tool
def tavily_search(
    # Required
    query: str,
    # Optional customisation parameters – all default to None which means "use heuristic defaults"
    max_results: int | None = None,
    search_depth: Literal["basic", "advanced"] | None = None,
    time_range: Literal["day", "week", "month", "year"] | None = None,
    topic: str | None = None,
    include_images: bool | None = None,
    include_image_descriptions: bool | None = None,
    include_raw_content: bool | None = None,
    include_domains: List[str] | None = None,
    exclude_domains: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Search the web using Tavily with fine-grained control.

    The function now supports additional optional parameters so that an LLM can
    explicitly influence the breadth or depth of the search when it has enough
    context to decide what it needs.

    If the optional arguments are **not** provided, the function falls back to a
    set of heuristic defaults that attempt to infer sensible search parameters
    from the query content (maintaining backwards-compatibility with the
    previous behaviour).
    """

    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    try:
        # Base search parameters from explicit user input (if any)
        search_params: Dict[str, Any] = {
            "query": query,
        }

        # Only populate a param if the caller supplied a non-None value. This
        # prevents us from overwriting the heuristic layer later on.
        if max_results is not None:
            search_params["max_results"] = max_results
        if search_depth is not None:
            search_params["search_depth"] = search_depth
        if time_range is not None:
            search_params["time_range"] = time_range
        if topic is not None:
            search_params["topic"] = topic
        if include_images is not None:
            search_params["include_images"] = include_images
        if include_image_descriptions is not None:
            search_params["include_image_descriptions"] = include_image_descriptions
        if include_raw_content is not None:
            search_params["include_raw_content"] = include_raw_content
        if include_domains is not None:
            search_params["include_domains"] = include_domains
        if exclude_domains is not None:
            search_params["exclude_domains"] = exclude_domains

        # ------------------------------------------------------------------
        # Heuristic defaults – applied **only** for parameters that the caller
        # did *not* specify explicitly above.
        # ------------------------------------------------------------------
        user_query = query.lower()

        def _set_default(key: str, value: Any) -> None:
            """Helper that sets a default only if the caller didn't override."""
            if key not in search_params:
                search_params[key] = value

        # Global sensible defaults
        _set_default("max_results", 10)
        _set_default("search_depth", "advanced")

        # Context-aware tweaks
        if any(word in user_query for word in ["news", "recent", "latest", "breaking"]):
            _set_default("topic", "news")
            _set_default("time_range", "week")
            _set_default("max_results", 8)
        elif any(
            word in user_query for word in ["stock", "market", "finance", "investment"]
        ):
            _set_default("topic", "finance")
            _set_default("time_range", "day")
            _set_default("exclude_domains", ["reddit.com", "twitter.com"])
        elif any(
            word in user_query for word in ["image", "picture", "visual", "photo"]
        ):
            _set_default("include_images", True)
            _set_default("include_image_descriptions", True)
            _set_default("max_results", 6)
        elif any(
            word in user_query for word in ["academic", "research", "study", "paper"]
        ):
            _set_default(
                "include_domains",
                [
                    "wikipedia.org",
                    "scholar.google.com",
                    "arxiv.org",
                ],
            )
            _set_default("include_raw_content", True)
            _set_default("max_results", 12)

        # Perform the search
        response = tavily_client.search(**search_params)

        # Normalise and return results
        return response.get("results", []) if isinstance(response, dict) else []

    except Exception as e:
        return [
            {
                "title": "Search Error",
                "url": "",
                "content": f"Error searching: {str(e)}",
                "score": 0.0,
            }
        ]


@tool
def tavily_extract_content(urls: List[str]) -> List[Dict[str, Any]]:
    """Extract content from web pages using Tavily.

    Args:
        urls: List of URLs to extract content from

    Returns:
        List of dictionaries containing extracted content with keys: url, content, images
    """
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    try:
        response = tavily_client.extract(
            urls=urls, include_images=False, extract_depth="advanced", format="markdown"
        )

        # Process the response and format it
        extracted_results = []
        if isinstance(response, dict) and "results" in response:
            for result in response["results"]:
                if isinstance(result, dict):
                    extracted_results.append(
                        {
                            "url": result.get("url", ""),
                            "content": result.get("raw_content", ""),
                            "images": result.get("images", []),
                        }
                    )

        return extracted_results

    except Exception as e:
        return [
            {
                "url": "error",
                "content": f"Error extracting content: {str(e)}",
                "images": [],
            }
        ]


@tool
def tavily_crawl(url: str) -> Dict[str, Any]:
    """Crawl a website starting from a base URL using Tavily.

    Args:
        url: The starting URL to crawl
        instructions: Natural language instructions for the crawler

    Returns:
        Dictionary containing crawl results with extracted content from multiple pages
    """
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    try:
        response = tavily_client.crawl(url, max_depth=1, max_breadth=20, limit=50)
        return (
            response
            if isinstance(response, dict)
            else {"error": "Invalid response format"}
        )
    except Exception as e:
        return {
            "base_url": url,
            "results": [],
            "error": f"Crawl failed: {str(e)}",
        }


@tool
def tavily_map_site(
    url: str,
    instructions: str = "",
    max_depth: int = 1,
    max_breadth: int = 20,
    limit: int = 50,
    select_paths: List[str] = None,
    select_domains: List[str] = None,
    exclude_paths: List[str] = None,
    exclude_domains: List[str] = None,
    allow_external: bool = False,
) -> Dict[str, Any]:
    """Map a website to get a sitemap starting from a base URL using Tavily.

    Args:
        url: The root URL to begin the mapping
        instructions: Natural language instructions for the crawler
        max_depth: Max depth of the mapping
        max_breadth: Max number of links to follow per level
        limit: Total number of links to process before stopping
        select_paths: Regex patterns to select only URLs with specific path patterns
        select_domains: Regex patterns to select crawling to specific domains
        exclude_paths: Regex patterns to exclude URLs with specific path patterns
        exclude_domains: Regex patterns to exclude specific domains
        allow_external: Whether to allow following links that go to external domains

    Returns:
        Dictionary containing base_url, results (list of discovered URLs), and response_time
    """
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    try:
        # Use correct parameter names based on official documentation
        map_params = {
            "url": url,
            "max_depth": max_depth,
            "max_breadth": max_breadth,
            "limit": limit,
            "allow_external": allow_external,
        }

        # Only add optional parameters if provided
        if instructions:
            map_params["instructions"] = instructions
        if select_paths:
            map_params["select_paths"] = select_paths
        if select_domains:
            map_params["select_domains"] = select_domains
        if exclude_paths:
            map_params["exclude_paths"] = exclude_paths
        if exclude_domains:
            map_params["exclude_domains"] = exclude_domains

        response = tavily_client.map(**map_params)
        return (
            response
            if isinstance(response, dict)
            else {"error": "Invalid response format"}
        )

    except Exception as e:
        # If map fails, fall back to search to discover URLs
        try:
            search_query = f"site:{url.split('//')[1].split('/')[0]}"
            if instructions:
                search_query += f" {instructions}"

            search_response = tavily_client.search(
                query=search_query, max_results=min(limit, 10), search_depth="basic"
            )

            # Extract URLs from search results
            discovered_urls = []
            for result in search_response.get("results", []):
                if result.get("url"):
                    discovered_urls.append(result["url"])

            return {
                "base_url": url,
                "results": discovered_urls,
                "response_time": search_response.get("response_time", 0.0),
                "fallback_used": "search",
                "original_error": f"Map failed: {str(e)}",
            }
        except Exception as search_error:
            return {
                "base_url": url,
                "results": [],
                "response_time": 0.0,
                "error": f"Both map and search failed. Map: {str(e)}, Search: {str(search_error)}",
            }
