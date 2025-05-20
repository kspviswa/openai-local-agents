import asyncio
import os

from openai import AsyncOpenAI

from agents import (
    Agent,
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
    set_tracing_disabled,
)

OLLAMA_MODEL = 'artifish/llama3.2-uncensored:latest'

# Using ModelProvider to create a custom model provider for Ollama

ollama_client = AsyncOpenAI(base_url='http://localhost:11434/v1', api_key='not needed')
set_tracing_disabled(disabled=True)

class OllamaProvider(ModelProvider):
    def get_model(self, model_name=OLLAMA_MODEL) -> Model:
        return OpenAIChatCompletionsModel(model=model_name, openai_client=ollama_client)

@function_tool
def get_system_time():
    print(f"[debug] Getting system time")
    return f"{os.popen('date').read()}"


async def main():
    agent = Agent(name="Assistant", instructions="You only respond as Mr Donald Trump. Max 2 lines.", tools=[get_system_time], model=OllamaProvider().get_model(OLLAMA_MODEL))

    # This will use the custom model provider
    result = await Runner.run(
        agent,
        "What time is it?",
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
