import os
import sqlite3
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.tools import WriteFileTool, ReadFileTool, ListDirectoryTool
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

load_dotenv()
prefix = "SILICON"
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


calculate = CalculateTool()
write_file = WriteFileTool()
read_file = ReadFileTool()
list_dir = ListDirectoryTool()
checkpoint_conn = sqlite3.connect("checkpoint.db", check_same_thread=False, isolation_level=None)
checkpointer = SqliteSaver(checkpoint_conn)

agent = create_agent(
    model=model,
    tools=[calculate, write_file, read_file, list_dir],
    system_prompt="你是一助手,会用工具计算，读写文件，列出目录。",
    debug=True,
    checkpointer=checkpointer,
    interrupt_before=["tools"]  # 工具节点执行前会暂停Agent。等待人工确认
)
config = {"configurable": {"thread_id": "session-2"}}

q = "计算2024*12+500"
response = agent.invoke({"messages": [{"role": "user", "content": q}]}, config=config)
print(f"Agent已暂停，最后消息：{response["messages"][-1].content[:100]}...")
user_input = input("确认执行？(yes/no): ")  #等待人工输入指令，人机协同，在关键操作前加入人工审批环节
if user_input == "yes":
    response = agent.invoke(Command(resume=user_input), config=config)
    for message in reversed(response["messages"]):   # 倒序查询第一条消息ToolMessage
        if isinstance(message, ToolMessage):
            print(f"答：{message.content}")
            break
else:
    print("已取消")
checkpoint_conn.close()
