from strands import Agent, AgentSkills, tool


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


plugin = AgentSkills(skills="skills/agent_skills")

research_agent = Agent(
    system_prompt="""You are a research assistant. Answer user questions using accurate, sourced information.

Guidelines:
- Use web_search to retrieve current information before answering.
- Do not infer, fabricate, or guess information not returned by tools.
- Cite the source of all factual claims.
""",
    tools=[web_search],
    plugins=[plugin],
)
