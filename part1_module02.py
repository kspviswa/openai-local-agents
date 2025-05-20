import asyncio
import os

from agents.extensions.models.litellm_provider import LitellmProvider
from agents.extensions.visualization import draw_graph

from agents import (
    Agent,
    Runner,
    function_tool,
    set_tracing_disabled,
)

OLLAMA_MODEL = 'artifish/llama3.2-uncensored:latest'
set_tracing_disabled(disabled=True)

# Using LitellmProvider to create a custom model provider for Ollama. But model name shold be prepended with ollama_chat/

@function_tool
def get_system_time():
    print(f"[debug] Getting system time")
    return f"{os.popen('date').read()}"


async def main():
    agent = Agent(name="Assistant",
                  instructions="You only respond as Mr Donald Trump. Max 2 lines.",
                  tools=[get_system_time],
                  model=LitellmProvider().get_model(f'ollama_chat/{OLLAMA_MODEL}'))

    result = await Runner.run(
        agent,
        "What time is it?",
    )
    print(result.final_output)

    draw_graph(agent, filename="graph")

if __name__ == "__main__":
    asyncio.run(main())
