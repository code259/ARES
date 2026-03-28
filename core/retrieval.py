from __future__ import annotations

import asyncio
import os
import xml.etree.ElementTree as ET

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
ARXIV_BASE = "https://export.arxiv.org/api/query"
OPENALEX_BASE = "https://api.openalex.org/works"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def semantic_scholar_search(query: str, limit: int = 5) -> list[dict]:
    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{SEMANTIC_SCHOLAR_BASE}/paper/search",
            params={
                "query": query,
                "limit": limit,
                "fields": "title,abstract,year,authors,externalIds,tldr,url",
            },
            headers=headers,
        )
        response.raise_for_status()
        return response.json().get("data", [])


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def arxiv_search(query: str, max_results: int = 5) -> list[dict]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            ARXIV_BASE,
            params={
                "search_query": f"all:{query}",
                "max_results": max_results,
                "sortBy": "relevance",
            },
        )
        response.raise_for_status()

    root = ET.fromstring(response.text)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    papers: list[dict] = []
    for entry in root.findall("atom:entry", namespace):
        papers.append(
            {
                "title": entry.findtext("atom:title", "", namespace).strip(),
                "abstract": entry.findtext("atom:summary", "", namespace).strip(),
                "id": entry.findtext("atom:id", "", namespace).strip(),
            },
        )
    return papers


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def openalex_search(query: str, limit: int = 5) -> list[dict]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            OPENALEX_BASE,
            params={
                "search": query,
                "per-page": limit,
                "select": "display_name,abstract_inverted_index,publication_year,doi,id",
            },
        )
        response.raise_for_status()
    results = response.json().get("results", [])
    papers: list[dict] = []
    for item in results:
        abstract_index = item.get("abstract_inverted_index") or {}
        if abstract_index:
            ordered = sorted(
                ((position, word) for word, positions in abstract_index.items() for position in positions),
                key=lambda pair: pair[0],
            )
            abstract = " ".join(word for _, word in ordered)
        else:
            abstract = ""
        papers.append(
            {
                "title": (item.get("display_name") or "").strip(),
                "abstract": abstract.strip(),
                "id": item.get("doi") or item.get("id") or "",
            },
        )
    return papers


async def fetch_papers_for_queries(
    queries: list[str],
    papers_per_query: int = 5,
) -> list[str]:
    seen_titles: set[str] = set()
    texts: list[str] = []

    async def fetch_one(query: str) -> list[dict]:
        combined: list[dict] = []
        try:
            combined.extend(await semantic_scholar_search(query, papers_per_query))
        except Exception:
            pass
        try:
            combined.extend(await openalex_search(query, papers_per_query))
        except Exception:
            pass
        try:
            combined.extend(await arxiv_search(query, papers_per_query))
        except Exception:
            pass
        return combined

    all_results = await asyncio.gather(*(fetch_one(query) for query in queries))
    for result_set in all_results:
        for paper in result_set:
            title = (paper.get("title") or "").strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            abstract = (paper.get("abstract") or paper.get("summary") or "").strip()
            texts.append(f"TITLE: {title}\n\nABSTRACT:\n{abstract}")

    return texts
