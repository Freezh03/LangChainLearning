# 导入必须的包
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 替换更适合中文的分割器
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import LangChainException

import os
import dotenv

# 加载环境变量
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

# 定义ChatDoc类，修复核心问题
class ChatDoc():
    def __init__(self, doc_path=None):
        self.doc_path = doc_path  # 文档路径
        self.split_text = []  # 分割后的文本（规范命名）
        self.vector_db = None  # 缓存向量库实例，避免重复创建
        # 修复提示词拼写错误，优化系统提示词
        self.template = [
            ("system",
             "你是一个专业的文档问答助手，仅基于提供的上下文内容回答问题，不编造信息。\n上下文内容：\n{context}\n"),
            ("human", "你好！"),
            ("ai", "您好，我是智能小智"),
            ("human", "{question}"),
        ]
        self.prompt = ChatPromptTemplate.from_messages(self.template)

    def validate_doc_path(self):
        """校验文档路径是否有效"""
        if not self.doc_path:
            raise ValueError("文档路径未设置，请先指定有效的doc_path")
        if not os.path.exists(self.doc_path):
            raise FileNotFoundError(f"文档不存在：{self.doc_path}")
        file_extension = self.doc_path.split(".")[-1]
        if file_extension != "docx":
            raise NotImplementedError(f"暂不支持{file_extension}格式，仅支持docx")
        return True

    def load_file(self):
        """加载docx文档（优化命名，增加异常处理）"""
        try:
            self.validate_doc_path()
            loader = Docx2txtLoader(self.doc_path)
            documents = loader.load()
            return documents
        except Exception as e:
            raise RuntimeError(f"加载文档失败：{str(e)}")

    def split_sentences(self):
        """分割文本（优化参数，适配中文）"""
        full_text = self.load_file()
        if not full_text:
            raise ValueError("文档内容为空，无法分割")
        # 替换为更适合中文的递归文本分割器，调整合理的chunk_size
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 增大chunk_size，保留完整上下文
            chunk_overlap=100,  # 重叠部分，保证上下文衔接
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "],  # 适配中文分隔符
            length_function=len,
            is_separator_regex=False
        )
        self.split_text = text_splitter.split_documents(full_text)
        if not self.split_text:
            raise ValueError("文本分割后为空，请检查文档内容")

    def init_vector_db(self, persist_directory="./asset/chroma-4"):
        """初始化向量库（仅执行一次，持久化存储）"""
        if not self.split_text:
            raise ValueError("请先调用split_sentences分割文本")
        if self.vector_db:  # 如果已初始化，直接返回
            return self.vector_db
        # 创建嵌入模型
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        # 持久化向量库到本地，避免重复创建
        self.vector_db = Chroma.from_documents(
            documents=self.split_text,
            embedding=embeddings,
            persist_directory=persist_directory  # 持久化路径
        )
        return self.vector_db

    def retrieve_context(self, question):
        """检索相关上下文（复用向量库，避免重复创建）"""
        if not self.vector_db:
            self.init_vector_db()  # 首次调用时初始化
        # 使用retriever检索相关文本
        retriever = self.vector_db.as_retriever(
            search_kwargs={"k": 3}  # 限制返回3个最相关的文本块，避免上下文过长
        )
        try:
            relevant_docs = retriever.invoke(question)
            return relevant_docs
        except Exception as e:
            raise RuntimeError(f"检索上下文失败：{str(e)}")

    def chat_with_doc(self, question):
        """与文档对话（核心方法，优化异常处理）"""
        # 1. 检索相关上下文
        relevant_docs = self.retrieve_context(question)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        print(f"检索到的上下文：\n{context}")

        # 2. 构建提示词
        try:
            messages = self.prompt.format_messages(context=context, question=question)
        except Exception as e:
            raise RuntimeError(f"构建提示词失败：{str(e)}")

        # 3. 调用LLM（增加超时和异常处理）
        try:
            llm = ChatOpenAI(
                model="gpt-4o-mini",  # 替换为更经济的模型，也可保留gpt-4
                temperature=0,
                timeout=30  # 增加超时限制
            )
            response = llm.invoke(messages)
            return response
        except LangChainException as e:
            raise RuntimeError(f"调用LLM失败：{str(e)}")

    def clear_vector_db(self):
        """清理向量库（可选，释放资源）"""
        if self.vector_db:
            self.vector_db.delete_collection()
            self.vector_db = None

# 测试代码
if __name__ == "__main__":
    try:
        # 初始化实例并指定文档路径
        chat_doc = ChatDoc(doc_path="C:/Users/huan.zheng/myproject/chapter07-Retrieval/asset/load/12-langchain.docx")
        # 分割文本（仅执行一次）
        chat_doc.split_sentences()
        # 首次提问会初始化向量库，后续提问复用
        response = chat_doc.chat_with_doc("LangChain的核心模块有哪些")
        print("\n问答结果：")
        print(response.content)

        # 第二次提问（复用向量库，性能提升）
        # response2 = chat_doc.chat_with_doc("LangChain如何实现文档分割")
        # print("\n第二次问答结果：")
        # print(response2.content)
    except Exception as e:
        print(f"程序执行失败：{str(e)}")
    finally:
        # 可选：清理向量库
        # chat_doc.clear_vector_db()
        pass