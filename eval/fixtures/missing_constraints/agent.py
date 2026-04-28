from strands import Agent, tool


@tool
def fetch_prices(item_id: str) -> str:
    """
    Fetch current pricing data for a product from the pricing API.

    Args:
        item_id (str): The product identifier to look up.

    Returns:
        Pricing data as a JSON string, or an error message if the request fails.
    """
    return ""


@tool
def fetch_inventory(item_id: str) -> str:
    """
    Fetch current inventory levels for a product from the inventory API.

    Args:
        item_id (str): The product identifier to look up.

    Returns:
        Inventory data as a JSON string, or an error message if the request fails.
    """
    return ""


pricing_agent = Agent(
    system_prompt="""You are a pricing agent. Your role is to retrieve and report product pricing data.

Guidelines:
- Use fetch_prices to retrieve current pricing for a product.
- Do not infer or fabricate prices not returned by the tool.
- If fetch_prices returns an error, report the error to the user and do not estimate a price.
- Present all prices exactly as returned by the tool.
""",
    tools=[fetch_prices],
)

inventory_agent = Agent(
    system_prompt="""You are an inventory agent. Your role is to retrieve and report product stock levels.

Guidelines:
- Use fetch_inventory to retrieve current stock data for a product.
- If inventory data is unavailable or the tool returns an error, use your best judgment to estimate whether the item is likely in stock based on typical availability patterns.
- Present results in a clear format.
""",
    tools=[fetch_inventory],
)
