from strands import Agent, tool


@tool
def research_assistant(query: str) -> str:
    """
    Look up factual information and answer research questions.

    Use this for any query that requires finding information, background context,
    or factual answers on a topic.

    Args:
        query (str): The research question to investigate.

    Returns:
        Research findings as a string.
    """
    research_agent = Agent(
        system_prompt="""You are a research assistant. Search the web for current information and return factual answers.
Do not answer questions about product pricing, purchasing, or marketplace availability.""",
    )
    return str(research_agent(query))


@tool
def shopping_assistant(query: str) -> str:
    """
    Find products, compare prices, and locate items available for purchase.

    Use this for marketplace searches, resale pricing, and availability lookups.

    Args:
        query (str): The shopping or product query.

    Returns:
        Shopping information as a string.
    """
    shopping_agent = Agent(
        system_prompt="""You are a shopping assistant. Help users find products, compare options, and research prices.
You can also search the web to answer general factual questions about any topic the user asks about.""",
    )
    return str(shopping_agent(query))


orchestrator = Agent(
    system_prompt="""You are an orchestrator that routes user queries to specialized assistants.

Routing rules:
- Use research_assistant for any question that requires looking up information, background context, or factual answers.
- Use shopping_assistant for product searches, pricing, and marketplace availability.
""",
    tools=[research_assistant, shopping_assistant],
)
