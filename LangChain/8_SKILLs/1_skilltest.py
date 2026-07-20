import os

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.tools import WriteFileTool, ReadFileTool, ListDirectoryTool
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

load_dotenv()
prefix = "SILICON"
# model = init_chat_model(
#     model_provider="openai",
#     configurable_fields=["model", "api_key", "base_url"],
#     config_prefix=prefix,
#     temperature=0.5,
#     max_tokens=1024,
# ).with_config({
#     "configurable": {
#         f"{prefix}_model": os.getenv(f"{prefix}_MODEL"),
#         f"{prefix}_api_key": os.getenv(f"{prefix}_API_KEY"),
#         f"{prefix}_base_url": os.getenv(f"{prefix}_BASE_URL"),
#     }
# })

# # deepagents 目前不支持init_chat_model构造的模型对象
# os.environ["OPENAI_API_KEY"] = os.getenv(f"{prefix}_API_KEY")
# os.environ["OPENAI_BASE_URL"] = os.getenv(f"{prefix}_BASE_URL")
# model = f"openai:{os.getenv(f'{prefix}_MODEL')}"

# return init_chat_model(model, **apply_provider_profile(model))    apply_provider_profile 对 "openai:" 前缀的模型注入了 output_version="responses/v1"，这导致 ChatOpenAI._use_responses_api() 返回 True，从而去调 OpenAI 的 /responses 端点。SiliconFlow 只兼容 Chat Completions API，不支持 Responses API —— 所以 404。
model =  ChatOpenAI(
    model= os.getenv(f"{prefix}_MODEL"),
    api_key=os.getenv(f"{prefix}_API_KEY"),
    base_url=os.getenv(f"{prefix}_BASE_URL"),
    use_responses_api=False
)

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

agent = create_deep_agent(
    model=model,
    tools=[calculate, write_file, read_file, list_dir],
    system_prompt="你是一助手,会用工具计算，读写文件，列出目录。",
    skills=["skills"],
    backend=FilesystemBackend(root_dir=os.getcwd(),virtual_mode=False),
    debug=True
)

queries = [
    "查询我的系统信息",
    "计算2024*12+500，然后把结果保存到result.txt",
    "读取 result.txt的内容",
    "列出当前目录文件",
]
for q in queries:
    print(f"\n问：{q}")
    result = agent.invoke({"messages":[{"role":"user","content":q}]})
    print(f"答：{result["messages"][-1].content}")

