import os

import sqlite3
from typing import Any
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.chat_models import init_chat_model
from langchain_community.tools import WriteFileTool, ReadFileTool, ListDirectoryTool
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langgraph.types import StateT
from langgraph.runtime import Runtime
from langgraph.typing import ContextT

load_dotenv()
prefix = "DASHSCOPE"
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

# 第一个middleware 用于记录模型的调用日志
class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> None: # 返回类型是None，因为这个Middleware只是记录日志，不会干预流程，所以直接返回None，告诉框架我完事了，继续往下走
        print(f"【日志】即将调用模型，当前消息数：{len(state['messages'])}")

    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> None:
        last_msg = state['messages'][-1]
        print(f"【日志】模型已响应{last_msg.content[:50]}...")

# 第二个middleware 用于安全检查，拦截危险操作
class SafetyMiddleware(AgentMiddleware):
    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:     # 不是单纯的None，因为这个middleware可能会干预流程
        last_msg = state['messages'][-1].content
        if "删除" in last_msg or "危险" in last_msg:
            return{
                "jump_to":"end",
                "messages": [AIMessage(content="检测到危险操作，已终止")]
            }
        return None


calculate = CalculateTool()
write_file = WriteFileTool()
read_file = ReadFileTool()
list_dir = ListDirectoryTool()
checkpoint_conn = sqlite3.connect("checkpoint.db", check_same_thread=False, isolation_level=None)
checkpointer = SqliteSaver(checkpoint_conn)
store_conn = sqlite3.connect("agent.db", check_same_thread=False, isolation_level=None)
store = SqliteStore(store_conn)

agent = create_agent(
    model=model,
    tools=[calculate, write_file, read_file, list_dir],
    system_prompt="你是一助手,会用工具计算，读写文件，列出目录。",
    debug=True,
    checkpointer=checkpointer,
    store=store,
    middleware=[LoggingMiddleware(), SafetyMiddleware()],
)
config = {"configurable": {"thread_id": "session-1"}}
store.put(
    ("user", "user-1"),
    "profile",
    {"name": "张三", "role": "developer", "skills": ["python", "typescript", "java"]}
)
profile = store.get(("user", "user-1"), "profile")
print(f"用户资料：{profile.value}")
queries = ["计算2024*12+500，然后把结果保存到result.txt",
           "读取 result.txt的内容",
           "列出当前目录文件",
           "刚才计算的结果是多少？"
           "删除所有文件"   # 触发安全拦截的提示词
           ]
for q in queries:
    print(f"\n问:{q}")
    response = agent.invoke({"messages": [{"role": "user", "content": q}]}, config=config)
    print(f"\n答：{response["messages"][-1].content}")

checkpoint_conn.close()
store_conn.close()
