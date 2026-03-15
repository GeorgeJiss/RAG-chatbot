from langchain_community.tools import DuckDuckGoSearchRun

def perform_web_search(query):
    """Performs a live web search using DuckDuckGo to get recent context."""
    try:
        search = DuckDuckGoSearchRun()
        result = search.invoke(query)
        return result
    except Exception as e:
        return f"Error performing web search: {str(e)}"
