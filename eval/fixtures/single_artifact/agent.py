from strands import Agent


analyst = Agent(
    system_prompt="""You are a data analyst. Your role is to retrieve and summarize structured data.

Do not infer or fabricate data not returned by tools.
""",
)
