from langchain_community.document_loaders import (
    TextLoader, CSVLoader, JSONLoader, PyPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# TextLoader
print("+++"*20)
txt_loader = TextLoader("load/01-langchain-utf-8.txt", encoding="utf-8")
txt_docs = txt_loader.load()
# for doc in txt_docs[:2]:
#     print(doc.page_content[:100])

print("+++"*20)
txt_loader = TextLoader("load/01-langchain-gbk.txt", encoding="gbk")
txt_docs = txt_loader.load()
# for doc in txt_docs[:2]:
#     print(doc.page_content[:100])

print("+++"*20)
# CSVLoader
csv_loader = CSVLoader("load/03-load.csv", encoding="gbk")
csv_docs = csv_loader.load()
# for doc in csv_docs[:2]:
#     print(doc.page_content[:100])

print("+++"*20)
# JSONLoader
json_loader = JSONLoader(file_path="load/04-load.json",
                         jq_schema=".employees[]",
                         text_content=False)
json_docs = json_loader.load()
# for doc in json_docs[:2]:
#     print(doc.page_content[:100])

print("+++"*20)
# Markdown
md_loader = TextLoader("load/06-load.md", encoding="utf-8")
md_docs = md_loader.load()
# for doc in md_docs[:2]:
#     print(doc.page_content[:100])


print("+++"*20)
# PyPDFLoader
pdf_loader = PyPDFLoader("load/02-load.pdf")
pdf_docs = pdf_loader.load()
# for doc in md_docs[:2]:
#     print(doc.page_content[:100])

all_docs = txt_docs + csv_docs + json_docs + md_docs + pdf_docs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    separators=["\n\n","。",",",""],
)
chunks=text_splitter.split_documents(all_docs)
print(f"原始文档数：{len(all_docs)}")
print(f"分割后的文本快数：{len(chunks)}")
for i, chunk in enumerate(chunks[:3]):
    print(f"块{i+1}{len(chunk.page_content)}字符：{chunk.page_content[:60]}")
    print(f"元数据：{chunk.metadata}")