# This file contains tools related to search.
from typing import Union, List, Dict, Any

from langchain_community.document_loaders import WikipediaLoader, WebBaseLoader, ArxivLoader
from langchain_core.tools import tool
from duckduckgo_search import DDGS


@tool
def web_search(query: str, max_results: int = 5) -> Dict[str, List[Dict[str, str]]]:
    """
    Search the web using DuckDuckGo and return a formatted list of results.

    Args:
        query: The search query.
        max_results: Number of results to retrieve (default 5).

    Returns:
        A dictionary with a single key 'web_results'.
        The value associated with 'web_results' is a list of dictionaries.
        Each dictionary represents a search result and contains:
        - 'title': The title of the search result (str).
        - 'url': The URL of the search result (str).
        - 'body': A snippet/summary of the search result content (str).
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            title = r.get("title", "")
            href = r.get("href", "")
            body = r.get("body", "")
            results.append({
                "title": title,
                "url": href,
                "body": body
            })

    return {"web_results": results}


@tool
def arxiv_search(query: str, max_results: int = 2) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search Arxiv for a query and return up to max_results documents.

    Args:
        query: The search query for Arxiv.
        max_results: Maximum number of search results to return (default is 2, 0 indexed).

    Returns:
        A dictionary with a single key 'arxiv_results'.
        The value associated with 'arxiv_results' is a list of dictionaries.
        Each dictionary represents an Arxiv document and contains detailed metadata (some of which can be optional) such as:
        - 'published': Publication date (str).
        - 'title': The title of the paper (str).
        - 'authors': List of authors comma seperated (str).
        - 'summary': Abstract of the paper (str).
        - 'content': Page content of the paper (str).
    """
    loader = ArxivLoader(query=query, load_max_docs=max_results)
    docs = loader.load()

    results = []
    for doc in docs:
        results.append({
            "published": doc.metadata.get("published", ""),
            "title": doc.metadata.get("Title", ""),
            "authors": doc.metadata.get("Authors", ""),
            "summary": doc.metadata.get("Summary", ""),
            "content": doc.page_content
        })

    return {"arxiv_results": results}


@tool
def wikipedia_search(query: str, load_max_docs: int = 3) -> Dict[str, List[Dict[str, str]]]:
    """
    Search Wikipedia for a query and return structured details of relevant articles.

    Args:
        query: The search query for Wikipedia.
        load_max_docs: Maximum number of Wikipedia articles to load (default is 3).

    Returns:
        A dictionary with a single key 'wiki_results'.
        The value associated with 'wiki_results' is a list of dictionaries.
        Each dictionary represents a Wikipedia article and contains:
        - 'source': The URL of the Wikipedia page (str).
        - 'summary': A brief summary of the article (str).
        - 'page': The title of the Wikipedia page (str).
        - 'content': The full text content of the page (str).
    """
    docs = WikipediaLoader(query=query, load_max_docs=load_max_docs).load()

    results = []
    for doc in docs:
        results.append({
            "source": doc.metadata.get("source", ""),
            "summary": doc.metadata.get("summary", ""),
            "page": doc.metadata.get("page", ""),
            "content": doc.page_content
        })

    return {"wiki_results": results}


@tool
def web_scraper(urls: Union[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    """
    Scrape full HTML content from one or more URLs.

    Args:
        urls: A single URL string or a list of URL strings to scrape.

    Returns:
        A dictionary where each key is a URL string.
        The value for each URL is another dictionary containing:
        - 'content': The full HTML content of the scraped page (str).
        - 'metadata': A dictionary of various metadata associated with the page,
                      which may include 'source', 'title', 'language', etc. (Dict[str, Any]).
    """
    # Ensure URLs is a list
    url_list = [urls] if isinstance(urls, str) else urls

    loader = WebBaseLoader(
        web_path=url_list,
        continue_on_failure=True,
        default_parser="html.parser"
    )
    docs = loader.load()

    results = {}
    for doc in docs:
        url = doc.metadata.get("source", "unknown_url")
        results[url] = {
            "content": doc.page_content,
            "metadata": doc.metadata
        }

    return results


if __name__ == '__main__':
    print(arxiv_search("clustering"))
