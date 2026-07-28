import asyncio
import os
import time
import dotenv
from langchain.chat_models import init_chat_model

# 加载环境变量
dotenv.load_dotenv()
prefix = "SILICON"
# 初始化大模型
try:
    chat_model = init_chat_model(
        model_provider="openai",
        configurable_fields=["model","api_key","base_url"],
        config_prefix=prefix,
    ).with_config({
        "configurable":{
            f"{prefix}_model":os.getenv(f"{prefix}_MODEL"),
            f"{prefix}_api_key":os.getenv(f"{prefix}_API_KEY"),
            f"{prefix}_base_url":os.getenv(f"{prefix}_BASE_URL"),            
        }
    })
    print(f"LLM初始化成功！")
except Exception as e:
    print(f"LLM初始化失败，失败原因：{e}")
    chat_model=None

# 同步调用（对比组）
async def demo_async_stream():
    print("+++演示：astream 的异步（非阻塞）效果+++")
    start_time = time.perf_counter()  # 记录开始时间
    print("程序开始...")

    # 1. 发起异步流式请求
    # 注意：此时请求已发出，返回的是一个异步生产器
    print(">>>发起异步模型调用（astream）...")
    stream_resp = chat_model.astream("请用一句话解释人工智能。")

    # 2. 在等待流式响应的同时，执行其他任务
    print(">>>流式请求已发送，程序无需等待，继续执行其他异步任务...")
    for i in range(3):
        # 使用 asyncio.sleep而非time.sleep
        # 这允许事件循环在等待时去处理上面的stream_resp网络 IO
        await asyncio.sleep(1)  # 异步等待，释放控制权
        print(f">>>正在执行第{i+1}个任务...(已耗时{time.perf_counter() - start_time:.2f}s)")

    # 3. 现在开始处理流式结果
    print(">>>模拟任务已完成，开始读取缓冲区的流式结果...")
    end_time = time.perf_counter()
    print(f">>>流式输出：",end="",flush=True)
    async for chunk in stream_resp:
        content=chunk.content if hasattr(chunk, 'content') else str(chunk)
        print(content, end="", flush=True)

    print("\n流式输出结束\n")
    print(f"++++总运行耗时：{end_time- start_time:.2f}s+++++")

# 入口：执行异步主函数
if __name__ == "__main__":
    asyncio.run(demo_async_stream())