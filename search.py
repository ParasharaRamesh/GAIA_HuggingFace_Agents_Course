'''
This file contains tools related to search.

Plan is to implement the following tools:

1. websearch tool
2. webscraper tool
3. arxiv/academic search tool
4. wikipedia search/scraper tool

TODO.1 handle chunking and storing in vector DB
TODO.2 handle exceptions , how does langgraph handle it?

'''
from typing import Union, List

from langchain_community.document_loaders import WikipediaLoader, WebBaseLoader
from langchain_core.tools import tool


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
    search_docs = WikipediaLoader(query=query, load_max_docs=load_max_docs).load()

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" summary="{doc.metadata.get("summary", "")}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"wiki_results": formatted_search_docs}


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
    pass
    # url = "https://blog.google/technology/google-deepmind/gemini-2-5-native-audio/"
    # results = web_scraper(url)