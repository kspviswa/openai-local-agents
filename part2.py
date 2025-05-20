import asyncio
import json
from util.ollamaProvider import OllamaProviderAsync
from agents import (
    Agent,
    ModelSettings,
    RunResult,
    Runner,
    function_tool,
    set_tracing_disabled,
)
from agents.extensions.visualization import draw_graph
set_tracing_disabled(disabled=True)

accounts_agent = Agent(
    name = "Accounts Agent",
    handoff_description="Specialist agent for customer accounts",
    instructions="You provide assistance with customer accounts. Answer questions about customer accounts.",
    model = OllamaProviderAsync().get_model(),
)

credit_card_agent = Agent(
    name = "Credit Card Agent",
    handoff_description="Specialist agent for customer credit cards",
    instructions="You provide assistance with customer credit cards. Answer questions about customer credit cards.",
    model = OllamaProviderAsync().get_model(),
)

@function_tool
def get_wire_transfer_status(order_id: str):
    """
    Retrieve the status of a wire transfer for a given order ID.

    Args:
        order_id (str): The unique identifier of the order.

    Returns:
        str: A message indicating the wire transfer status for the specified order ID.
    """
    print(f"[debug] Getting wire transfer status")
    return f"Wire transfer status for {order_id}: Completed"

wire_transfer_agent = Agent(
    name = "Wire Transfer Agent",
    handoff_description="Specialist agent for customer wire transfers",
    instructions="You provide assistance with customer wire transfers. Answer questions about customer wire transfers.",
    model = OllamaProviderAsync().get_model(),
    tools=[get_wire_transfer_status],
    tool_use_behavior="stop_on_first_tool"
)

main_agent = Agent(
    name = "Operator Agent",
    instructions="You determine which agent to use based on the user's question. If the question is about customer accounts, credit cards, or wire transfers, hand off to the appropriate agent. Otherwise, answer the question yourself.",
    model = OllamaProviderAsync().get_model(),
    handoffs=[accounts_agent, credit_card_agent, wire_transfer_agent],
    model_settings=ModelSettings(tool_choice="none")
)

draw_graph(main_agent, filename="graph3")

async def main():
    print("Enter your question (or 'exit' to quit): ")
    while True:
        user_input = input("User => ")
        if user_input.lower() == 'exit':
            break
        result: RunResult = await Runner.run(main_agent, user_input)
        if result and result.last_agent in [accounts_agent, credit_card_agent, wire_transfer_agent]:
            print(f'[debug] Handoff to {result.last_agent.name}')
            sub_result: RunResult = await Runner.run(result.last_agent, user_input)
            print(f"Assistant [{result.last_agent.name}] => {sub_result.final_output}")
        elif result.final_output:
            print(f"Assistant [{result.last_agent.name}] => {result.final_output}")

asyncio.run(main())