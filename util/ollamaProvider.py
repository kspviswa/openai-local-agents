
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel, Model, ModelProvider

OLLAMA_MODEL = 'artifish/llama3.2-uncensored:latest'
#OLLAMA_MODEL = 'qwen3'
#OLLAMA_MODEL = 'llama3.2'

# Using ModelProvider to create a custom model provider for Ollama

ollama_client = AsyncOpenAI(base_url='http://localhost:11434/v1', api_key='not needed')

class OllamaProviderAsync(ModelProvider):
    def get_model(self, model_name=OLLAMA_MODEL) -> Model:
        return OpenAIChatCompletionsModel(model=model_name, openai_client=ollama_client)
