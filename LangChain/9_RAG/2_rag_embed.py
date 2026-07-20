from langchain_community.document_loaders import (
    TextLoader, CSVLoader, JSONLoader, PyPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

# TextLoader
txt_loader = TextLoader("load/01-langchain-utf-8.txt", encoding="utf-8")
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
md_loader = TextLoader("load/06-load.md", encoding="utf-8")
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
    cache_dir=r"D:\myprojects\LangChainLearning\LangChain\9_RAG\model"
)

model = SentenceTransformer(model_dir)
chunk_texts = [chunk.page_content for chunk in chunks if chunk.page_content.strip()]
vectors = model.encode(chunk_texts)
print(f"文本块数量：{len(chunks)}")
print(f"向量数量：{len(vectors)}")
print(f"每个向量的维度：{len(vectors[0])}")


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_top_similar(query_text, vectors, chunk_texts, topn_n=3):
    query_vector = model.encode([query_text])[0]
    similarities = []
    for i, v in enumerate(vectors):
        score = cosine_similarity(query_vector, v)
        similarities.append((i, score))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:topn_n]
   

for query in ["LangChain是一个AI应用开发框架", "是大语言模型应用的开发工具"]:
    print(f"\n'{query}'与各文本块的相似度（Top 3）：")
    for idx, score in get_top_similar(query, vectors, chunk_texts):
        print(f"相似度{score:.4f}:{chunk_texts[idx][:60]}")
