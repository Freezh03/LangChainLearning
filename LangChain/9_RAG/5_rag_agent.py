import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import (
    TextLoader, CSVLoader, JSONLoader, PyPDFLoader
)
from langchain_community.tools import WriteFileTool, ReadFileTool, ListDirectoryTool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from modelscope.ops.image_control_3d_portrait.torch_utils.persistence import persistent_class
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_chroma import Chroma
from langchain_core.tools import create_retriever_tool, BaseTool

# TextLoader
txt_loader = TextLoader("load/clothes.txt", encoding="utf-8")
txt_docs = txt_loader.load()

# CSVLoader
csv_loader = CSVLoader("load/03-load.csv", encoding="gbk")
csv_docs = csv_loader.load()

# JSONLoader
json_loader = JSONLoader(file_path="load/04-load.json",
                         jq_schema=".employees[]",
                         text_content=False)
json_docs = json_loader.load()

# Markdown
md_loader = TextLoader("load/2025report.md", encoding="utf-8")
md_docs = md_loader.load()

# PyPDFLoader
pdf_loader = PyPDFLoader("load/02-load.pdf")
pdf_docs = pdf_loader.load()

all_docs = txt_docs + csv_docs + json_docs + md_docs + pdf_docs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    separators=["\n\n", "。", ",", ""],
)

# 对所有文档进行分割
chunks = text_splitter.split_documents(all_docs)

# 模型下载重载
from modelscope import snapshot_download

model_dir = snapshot_download(
    model_id="BAAI/bge-small-zh-v1.5",
    cache_dir=r"C:\Users\huan.zheng\PycharmProjects\LangChainLearning\LangChain\9_RAG\model"
)

model_rag = SentenceTransformer(model_dir)
chunk_texts = [chunk.page_content for chunk in chunks if chunk.page_content.strip()]
vectors = model_rag.encode(chunk_texts)
print(f"文本块数量：{len(chunks)}")
print(f"向量数量：{len(vectors)}")
print(f"每个向量的维度：{len(vectors[0])}")


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_top_similar(query_text, vectors, chunk_texts, topn_n=3):
    query_vector = model_rag.encode([query_text])[0]
    similarities = []
    for i, v in enumerate(vectors):
        score = cosine_similarity(query_vector, v)
        similarities.append((i, score))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:topn_n]


persist_directory = "chroma_study_db"


class LocalEmbedding:
    def embed_documents(self, texts):
        return model_rag.encode(texts).tolist()

    def embed_query(self, text):
        return model_rag.encode([text])[0].tolist()


embedding_function = LocalEmbedding()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_function,
    persist_directory=persist_directory
)
print(f"向量库存储条数：{vectorstore._collection.count()}")

query = "有没有简约通勤纯棉长袖衬衫?"
results = vectorstore.similarity_search(query, k=2)
for i, doc in enumerate(results):
    print(f"{i + 1}.{doc.page_content[:80]}")
    print(f"元数据：{doc.metadata}")
results_mmr = vectorstore.max_marginal_relevance_search(
    query,
    k=2,
    fetch_k=10,
    lambda_mult=0.7
)
for i, doc in enumerate(results_mmr):
    print(f"{i + 1}.{doc.page_content[:80]}")
results_filtered = vectorstore.similarity_search(
    "长袖衬衫",
    k=2,
    filter={"gender": "女"}
)

for doc in results_filtered:
    print(f"{doc.page_content[:80]}")
retrieve = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)

load_dotenv()
prefix = "SILICONFLOW"
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
search_tool = create_retriever_tool(
    retriever=retrieve,
    name="search",
    description="搜索衣服产品信息，当用户询问衣服相关问题时使用此工具"
)

agent = create_agent(
    model=model,
    tools=[calculate, write_file, read_file, list_dir, search_tool],
    system_prompt="你是一助手,可以搜索衣服相关的信息，会用工具计算，读写文件，列出目录。",
    debug=True
)
queries = [
    "你好",
    "有没有简约通勤纯棉长袖衬衫?",
    "计算2024*12+500，然后把结果保存到result.txt",
    "读取 result.txt的内容",
    "列出当前目录文件"]
for q in queries:
    print(f"\n问:{q}")
    response = agent.invoke({"messages": [{"role": "user", "content": q}]})
    print(f"\n答：{response["messages"][-1].content}")
