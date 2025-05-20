import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
import json

SSE_URL = "http://localhost:8000/sse"

async def main():
    async with sse_client(SSE_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {[tool.name for tool in tools.tools]}")
            
            # Call get_current_weather tool if available
            if any(tool.name == "get_current_weather" for tool in tools.tools):
                weather = await session.call_tool("get_current_weather", {"location": "Ottawa, ON, Canada"})
                print(f"🌤 Weather Update: {weather.content[0].text}")
            else:
                print("Weather service unavailable")

if __name__ == "__main__":
    asyncio.run(main())
