from agents import Agent, InputGuardrail, GuardrailFunctionOutput, InputGuardrailTripwireTriggered, Runner, function_tool, set_tracing_disabled
from agents.extensions.visualization import draw_graph
from pydantic import BaseModel
import asyncio
from util.ollamaProvider import OllamaProviderAsync
from util.localRAGProvider import LocalRAGProvider

set_tracing_disabled(disabled=True)

# Initialize the LocalRAGProvider
ollama_rag = LocalRAGProvider(model_name="nomic-embed-text:latest", chunk_size=1000, chunk_overlap=200)
ollama_rag.load_documents("./kb/")

# Define a Pydantic model for the Input Guardrail response
class InputGuardrailResponse(BaseModel):
    isValidQuestion: bool
    reasoning: str
    polite_decline_response: str

# Define the RAG Tool
@function_tool
def get_answer_about_pydanticai(query: str) -> str:
    """
    Retrieve an answer about PydanticAI from the local RAG provider.
    
    Args:
        query (str): The question to ask.
    
    Returns:
        str: The answer to the question.
    """
    print(f"[debug] Getting answer for query: {query}")
    return ollama_rag.query(query)


# Define the InputGuardrail Agent
input_guardrail_agent = Agent(
    name="Input Guardrail Agent",
    instructions="Check whether the input is a valid question about PydanticAI. If you determine the input is invalid, provide a polite decline response stating that you can only answer questions about PydanticAI. Don't attempt to provide answer to the question.",
    model=OllamaProviderAsync().get_model(),
    handoffs=[],
    output_type=InputGuardrailResponse
)

# Define the InputGuardrail function
async def pydanticai_input_guardrail(ctx, agent,input_text):
    result = await Runner.run(input_guardrail_agent, input_text)
    final_result = result.final_output_as(InputGuardrailResponse)
    return GuardrailFunctionOutput(
        output_info=result,
        tripwire_triggered= not final_result.isValidQuestion,
    )

# Define the RAG Agent
rag_agent = Agent(
    name="RAG Agent",
    instructions="You are an expert answering questions about Pydanti AI. Use the tools at your disposal to answer the question.",
    model=OllamaProviderAsync().get_model(),
    handoffs=[],
    tools=[get_answer_about_pydanticai],
    output_type=str,
    input_guardrails=[
        InputGuardrail(guardrail_function=pydanticai_input_guardrail)
    ]
)

async def main():
    try:
        #result = await Runner.run(rag_agent, "What is PydanticAI?")
        result = await Runner.run(rag_agent, "How do I install pytorch?")
        print(f"Result: {result.final_output}")
    except InputGuardrailTripwireTriggered as e:
        print(f"[debug] Tripwire triggered: {e.guardrail_result.output.output_info.final_output.polite_decline_response}")
    draw_graph(rag_agent, filename="graph4")

if __name__ == "__main__":
    asyncio.run(main())