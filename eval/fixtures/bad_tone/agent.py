from strands import Agent, tool


@tool
def search_web(query: str) -> str:
    """
    hey just use this to search the web for whatever you need lol

    Args:
        query (str): whatever you want to look up

    Returns:
        some results hopefully
    """
    return ""


@tool
def summarize_text(text: str) -> str:
    """
    takes some text and makes it shorter, pretty simple

    Args:
        text (str): the stuff to summarize

    Returns:
        shorter version of the input or whatever
    """
    return ""


researcher = Agent(
    system_prompt="you're a research helper, just find stuff and answer things, idk do your best :)",
)
