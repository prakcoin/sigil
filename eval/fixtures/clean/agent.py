from strands import Agent, tool


@tool
def web_search(query: str) -> str:
    """
    Search the web for factual information on a topic.

    Use this tool when the query requires current or external information not available in context.

    Args:
        query (str): The search query string.

    Returns:
        Search results as a JSON string. Returns an error message prefixed with 'Error:' if the request fails.
    """
    return ""


@tool
def summarize(text: str) -> str:
    """
    Condense a long text into a concise summary.

    Use this after retrieving content that needs to be shortened before presenting to the user.

    Args:
        text (str): The full text to condense.

    Returns:
        A concise summary string preserving key facts and conclusions.
    """
    return ""


research_agent = Agent(
    system_prompt="""You are a research assistant. Your role is to answer user questions using accurate, sourced information.

Guidelines:
- Use web_search to retrieve current information before answering.
- Do not infer, fabricate, or guess information not returned by tools.
- Cite the source of all factual claims.
- Keep responses concise and factual.
- If a tool returns an error, surface the error to the user and do not attempt to fill in missing data.
""",
    tools=[web_search, summarize],
)
