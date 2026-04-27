from strands import Agent, tool


@tool
def analyze_document(text: str) -> str:
    """
    Utilize this tool in order to perform document analysis.

    Use this when you need to leverage text analysis capabilities to extract key information.

    Args:
        text (str): The document text to be processed and analyzed.

    Returns:
        An analysis of the document's key themes and structure.
    """
    return ""


@tool
def generate_report(findings: str) -> str:
    """
    Generate a structured report utilizing the provided findings.

    Leverage this tool to transform raw findings into a formatted output.

    Args:
        findings (str): The raw analysis findings to incorporate.

    Returns:
        A formatted report string.
    """
    return ""


analyst = Agent(
    system_prompt="""You are an analyst. Utilize your tools in order to provide data-driven insights.
Leverage the analyze_document tool to process inputs, then utilize generate_report to present results.""",
)
