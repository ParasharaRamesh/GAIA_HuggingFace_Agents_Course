'''
This file contains tools related to search.

Plan is to implement the following tools:

1. websearch tool
2. webscraper tool
3. arxiv/academic search tool
4. wikipedia search/scraper tool

TODO.0 return only a part of the chunk (example summary if present) to the LLM context window while indexing the rest into the DB
TODO.1 handle chunking and storing in vector DB
TODO.2 handle exceptions , how does langgraph handle it?

'''
from typing import Union, List

from langchain_community.document_loaders import WikipediaLoader, WebBaseLoader, ArxivLoader
from langchain_core.tools import tool
from duckduckgo_search import DDGS


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo and return a formatted list of results.

    Args:
        query: The search query.
        max_results: Number of results to retrieve (default 5).

    Returns:
        A formatted string of search results.
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
def arxiv_search(query: str, max_results: int = 2) -> dict:
    """
    Search Arxiv for a query and return up to max_results documents.(default is 2)

    Args:
        query: The search query.
        max_results: Maximum number of search results to return.

    Returns:
        A dictionary containing the search results, each with source URL, summary, and page content.
    """
    loader = ArxivLoader(query=query, load_max_docs=max_results)
    docs = loader.load()

    results = []
    for doc in docs:
        results.append({
            "source": doc.metadata.get("source", ""),
            "summary": doc.metadata.get("summary", ""),
            "page": doc.metadata.get("page", ""),
            "content": doc.page_content
        })

    return {"arxiv_results": results}


@tool
def wiki_search(query: str, load_max_docs: int = 3) -> str:
    """
    Search Wikipedia for a query and return maximum 3 results with page content.

    Args:
        query: The search query.
        load_max_docs: The maximum number of documents to load. Default is 3 (good enough for most tasks)

    Returns:
        A formatted string containing the source URL, summary and page content for the related/relevant articles based on the 'load_max_docs' parameter
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
def web_scraper(urls: Union[str, List[str]]) -> dict:
    """
    Scrape full HTML content from one or more URLs.

    Args:
        urls: A single URL string or a list of URL strings.

    Returns:
        A dictionary mapping each URL to its scraped HTML content and metadata.
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
            "html_content": doc.page_content,
            "metadata": doc.metadata
        }

    return results


if __name__ == '__main__':
    print(web_search("sanju samson"))
