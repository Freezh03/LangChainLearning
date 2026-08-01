import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from rich import print as rprint
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler


load_dotenv(r"C:\Users\huan.zheng\myproject\LangChainLearning\.env")

# 1. 初始化Langfuse，自动读.env变量
langfuse = Langfuse()
# 生成LangChain回调处理器
lf_handler = CallbackHandler()

llm = init_chat_model(
    model="zai-org/GLM-5.2",
    model_provider="openai",
    api_key=os.getenv("SILICON_API_KEY"),
    base_url=os.getenv("SILICON_BASE_URL"),
    temperature=0.5,
    max_tokens=200,
    # 指定可调整的参数
    configurable_fields=["model","model_provider","temperature","max_tokens"],
)

config = {
    "run_name":"joke_generation",    # 在LangSmith中这次运行会显示为"joke_generation"
    "tags":["my_tag1","my_tag2"],          # 打上标签便于分类查找
    "metadata":{
        "user_id":"victorzh",        # 记录用户ID
        "session_id":"sess_123"      # 记录会话ID
    },    
    "configurable":{
        "model":"deepseek-ai/DeepSeek-V4-Flash",   # 配置模型参数
        "model_provider":"deepseek", # 配置模型提供商
        "temperature":0.7,           # 配置温度参数
        "max_tokens":1000            # 配置最大令牌数
    },
    "callbacks":[lf_handler]
}

# 调用模型并传入config
res = llm.invoke("给我讲一个关于打雷的笑话",config=config)
rprint(res)