import os

import asyncio
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.tools import WriteFileTool, ReadFileTool, ListDirectoryTool
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()
prefix = "QWEN"
model = init_chat_model(
    model_provider="openai",
    configurable_fields=["model", "api_key", "base_url"],
    config_prefix=prefix,
    temperature=0.5,
    max_tokens=1024,
).with_config({
    "configurable": {
        f"{prefix}_model": os.getenv(f"{prefix}_MODEL"),
        f"{prefix}_api_key": os.getenv(f"{prefix}_API_KEY"),
        f"{prefix}_base_url": os.getenv(f"{prefix}_BASE_URL"),
    }
})


class CalculateTool(BaseTool):
    name: str = "calculate"
    description: str = "计算数学表达式的值"

    def _run(self, expression: str) -> str:
        try:
            return f"计算结果：{eval(expression)}"
        except Exception as e:
            return f"计算错误:{str(e)}"

    async def _arun(self, expression: str) -> str:
        return self._run(expression)


async def main():
    client = MultiServerMCPClient({
        "weather": {"url": "http://127.0.0.1:8080/mcp",
                    "transport": "streamable_http"}
    })
    mcp_tools = await client.get_tools()
    calculate = CalculateTool()
    write_file = WriteFileTool()
    read_file = ReadFileTool()
    list_dir = ListDirectoryTool()

    agent = create_agent(
        model=model,
        tools=[calculate, write_file, read_file, list_dir] + mcp_tools,
        system_prompt="你是一助手,会用工具计算，读写文件，列出目录。",
        debug=True,
    )

    queries = [
        "北京天气怎么样？",
        "计算 2024*12+500,然后把结果保存到 result.txt",
        "读取 result.txt",
        "列出当前目录文件"
    ]

    for q in queries:
        print(f"\n问：{q}")
        result = await agent.ainvoke({"messages": [{"role": "user", "content": q}]})
        print(f"答：{result['messages'][-1].content}")


asyncio.run(main())
