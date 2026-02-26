"""Web tools: web_search and web_fetch."""

import html
import json
import os
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from nanobot.agent.tools.base import Tool
from nanobot.utils.token_tracker import track_model_token_usage

# Shared constants
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"
MAX_REDIRECTS = 5  # Limit redirects to prevent DoS attacks


def _strip_tags(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r'<script[\s\S]*?</script>', '', text, flags=re.I)
    text = re.sub(r'<style[\s\S]*?</style>', '', text, flags=re.I)
    text = re.sub(r'<[^>]+>', '', text)
    return html.unescape(text).strip()


def _normalize(text: str) -> str:
    """Normalize whitespace."""
    text = re.sub(r'[ \t]+', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate URL: must be http(s) with valid domain."""
    try:
        p = urlparse(url)
        if p.scheme not in ('http', 'https'):
            return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
        if not p.netloc:
            return False, "Missing domain"
        return True, ""
    except Exception as e:
        return False, str(e)


class WebSearchTool(Tool):
    """Search the web using multiple providers (Brave, Perplexity, Serper, Tavily)."""
    
    name = "web_search"
    description = "Search the web. Returns titles, URLs, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Results (1-10)", "minimum": 1, "maximum": 10}
        },
        "required": ["query"]
    }
    
    def __init__(self, api_key: str | None = None, max_results: int = 5, provider: str = "brave"):
        self.provider = provider.lower()
        self.api_key = api_key or self._get_env_api_key(self.provider)
        self.max_results = max_results
    
    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        if not self.api_key:

            return f"Error: API key not configured for provider '{self.provider}'"

        n = min(max(count or self.max_results, 1), 10)

        if self.provider == "brave":
            return await self._brave(query, n)
        if self.provider == "perplexity":
            return await self._perplexity(query, n)
        if self.provider == "serper":
            return await self._serper(query, n)
        if self.provider == "tavily":
            return await self._tavily(query, n)

        return f"Error: Unknown provider '{self.provider}'"

    def _get_env_api_key(self, provider: str) -> str:
        env_map = {
            "brave": "BRAVE_API_KEY",
            "perplexity": "PERPLEXITY_API_KEY",
            "serper": "SERPER_API_KEY",
            "tavily": "TAVILY_API_KEY",
        }
        return os.environ.get(env_map.get(provider, ""), "")

    async def _brave(self, query: str, count: int) -> str:
        """Brave Search API."""
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": count},
                    headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
                    timeout=10.0,
                )
                r.raise_for_status()

            results = r.json().get("web", {}).get("results", [])
            return self._format_results(query, results, lambda x: {
                "title": x.get("title", ""),
                "url": x.get("url", ""),
                "snippet": x.get("description", ""),
            })
        except Exception as e:
            return f"Error: {e}"

    async def _perplexity(self, query: str, count: int) -> str:
        """Perplexity API using online model with citations."""
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "sonar",
                        "messages": [
                            {"role": "user", "content": query}
                        ],
                        "max_tokens": 1024
                    },
                    timeout=30.0,
                )
                r.raise_for_status()

            data = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            usage = data.get("usage") if isinstance(data, dict) else None
            if isinstance(usage, dict):
                try:
                    track_model_token_usage(model="perplexity/sonar", usage=usage)
                except Exception:
                    # Web search should remain available even if usage tracking fails.
                    pass
            
            # Extract citations from the response
            citations = []
            if "citations" in data:
                citations = data.get("citations", [])
            elif "choices" in data and len(data["choices"]) > 0:
                # Try to extract citations from message content
                message = data["choices"][0].get("message", {})
                if "citations" in message:
                    citations = message.get("citations", [])

            result = f"Results for: {query}\n\n{content}" if content else f"No results for: {query}"
            if citations:
                result += "\n\nSources:\n" + "\n".join(
                    f"{i + 1}. {url}" for i, url in enumerate(citations[:count])
                )
            return result
        except Exception as e:
            return f"Error: {e}"

    async def _serper(self, query: str, count: int) -> str:
        """Serper API."""
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                    json={"q": query, "num": count},
                    timeout=10.0,
                )
                r.raise_for_status()

            results = r.json().get("organic", [])
            return self._format_results(query, results, lambda x: {
                "title": x.get("title", ""),
                "url": x.get("link", ""),
                "snippet": x.get("snippet", ""),
            })
        except Exception as e:
            return f"Error: {e}"

    async def _tavily(self, query: str, count: int) -> str:
        """Tavily Search API."""
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    "https://api.tavily.com/search",
                    headers={"Content-Type": "application/json"},
                    json={"api_key": self.api_key, "query": query, "max_results": count},
                    timeout=10.0,
                )
                r.raise_for_status()

            results = r.json().get("results", [])
            return self._format_results(query, results, lambda x: {
                "title": x.get("title", ""),
                "url": x.get("url", ""),
                "snippet": x.get("content", ""),
            })
        except Exception as e:
            return f"Error: {e}"

    def _format_results(self, query: str, results: list[dict[str, Any]], mapper: Any) -> str:
        """Format search results consistently."""
        if not results:
            return f"No results for: {query}"

        lines = [f"Results for: {query}\n"]
        for i, item in enumerate(results, 1):
            mapped = mapper(item)
            lines.append(f"{i}. {mapped['title']}\n   {mapped['url']}")
            if mapped.get("snippet"):
                lines.append(f"   {mapped['snippet']}")
        return "\n".join(lines)


class WebFetchTool(Tool):
    """Fetch and extract content from a URL using Readability."""
    
    name = "web_fetch"
    description = "Fetch URL and extract readable content (HTML → markdown/text)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extractMode": {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            "maxChars": {"type": "integer", "minimum": 100}
        },
        "required": ["url"]
    }
    
    def __init__(self, max_chars: int = 50000):
        self.max_chars = max_chars
    
    async def execute(self, url: str, extractMode: str = "markdown", maxChars: int | None = None, **kwargs: Any) -> str:
        from readability import Document

        max_chars = maxChars or self.max_chars

        # Validate URL before fetching
        is_valid, error_msg = _validate_url(url)
        if not is_valid:
            return json.dumps({"error": f"URL validation failed: {error_msg}", "url": url}, ensure_ascii=False)

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=MAX_REDIRECTS,
                timeout=30.0
            ) as client:
                r = await client.get(url, headers={"User-Agent": USER_AGENT})
                r.raise_for_status()
            
            ctype = r.headers.get("content-type", "")
            
            # JSON
            if "application/json" in ctype:
                text, extractor = json.dumps(r.json(), indent=2, ensure_ascii=False), "json"
            # HTML
            elif "text/html" in ctype or r.text[:256].lower().startswith(("<!doctype", "<html")):
                doc = Document(r.text)
                content = self._to_markdown(doc.summary()) if extractMode == "markdown" else _strip_tags(doc.summary())
                text = f"# {doc.title()}\n\n{content}" if doc.title() else content
                extractor = "readability"
            else:
                text, extractor = r.text, "raw"
            
            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]
            
            return json.dumps({"url": url, "finalUrl": str(r.url), "status": r.status_code,
                              "extractor": extractor, "truncated": truncated, "length": len(text), "text": text}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e), "url": url}, ensure_ascii=False)
    
    def _to_markdown(self, html: str) -> str:
        """Convert HTML to markdown."""
        # Convert links, headings, lists before stripping tags
        text = re.sub(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a>',
                      lambda m: f'[{_strip_tags(m[2])}]({m[1]})', html, flags=re.I)
        text = re.sub(r'<h([1-6])[^>]*>([\s\S]*?)</h\1>',
                      lambda m: f'\n{"#" * int(m[1])} {_strip_tags(m[2])}\n', text, flags=re.I)
        text = re.sub(r'<li[^>]*>([\s\S]*?)</li>', lambda m: f'\n- {_strip_tags(m[1])}', text, flags=re.I)
        text = re.sub(r'</(p|div|section|article)>', '\n\n', text, flags=re.I)
        text = re.sub(r'<(br|hr)\s*/?>', '\n', text, flags=re.I)
        return _normalize(_strip_tags(text))
