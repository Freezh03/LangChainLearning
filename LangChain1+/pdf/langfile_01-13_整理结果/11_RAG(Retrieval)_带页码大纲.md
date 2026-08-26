# 11_RAG(Retrieval) 大纲

# 1、Retrieval模块的设计意义（第 1 页）

## 1.1 大模型的局限（第 1 页）

### 1）知识滞后（第 1 页）

### 2）知识缺失（第 1 页）

### 3）幻觉（第 1 页）

## 1.2 什么是RAG（第 1 页）

## 1.3 RAG优缺点（第 4 页）

### RAG的优点（第 4 页）

### RAG的缺点（第 4 页）

## 1.4 RAG工作流程（第 4 页）

### 环节1：Source（数据源）（第 4 页）

### 环节2：Load（加载）（第 5 页）

### 环节3：Transform（转换）（第 5 页）

#### 环节3.1：Text Splitting（文档拆分）（第 5 页）

### 环节4：Embed（嵌入）（第 6 页）

### 环节5：Store（存储）（第 7 页）

### 环节6：Retrieve（检索）（第 7 页）

# 2、详细使用流程（第 7 页）

## 2.1 环境准备（第 7 页）

### 2.1.1 安装依赖（第 7 页）

### 2.1.2 准备数据（第 8 页）

## 2.2 文档加载器 Document Loaders（第 8 页）

### 2.2.1 加载txt（第 8 页）

### 2.2.2 加载CSV（第 9 页）

### 2.2.3 加载JSON（第 10 页）

#### 举例1：使用JSONLoader文档加载器加载（第 10 页）

#### 举例2：提取04-response.json文件中指定的文本（第 11 页）

##### 加载json文件中所有的数据（第 11 页）

##### 加载json文件中employees[]中的name字段（第 11 页）

##### 提取04-response.json文件中嵌套在data.items[].content的文本（第 12 页）

##### 提取04-response.json文件中嵌套在data.items[]里的title、content和其文本（第 13 页）

### 2.2.4 加载pdf（第 13 页）

#### 方式1：PyPDFLoader（第 14 页）

#### 方式2：MinerU（第 16 页）

### 2.2.5 加载word（第 18 页）

### 2.2.6 加载Markdown（第 19 页）

#### 举例1：使用UnstructuredMarkdownLoader加载md文（第 19 页）

#### 举例2：精细分割文档，保留结构信息（第 20 页）

### 2.2.7 加载HTML(了解)（第 21 页）

### 2.2.8 加载File Directory(了解)（第 24 页）

### 2.2.9 了解：BaseLoader、Document类（第 25 页）

#### BaseLoader类分析（第 25 页）

#### Document类分析（第 26 页）

## 2.3 文档切分器 Text Splitters（第 26 页）

### 2.3.1 为什么分割/切分/分块？（第 26 页）

### 2.3.2 Chunking拆分的策略（第 27 页）

#### 方法1：根据句子切分（第 27 页）

#### 方法2：按照固定字符数来切分（第 27 页）

#### 方法3：按固定字符数来切分，结合重叠窗口（overlapping windows）（第 27 页）

#### 方法4：递归字符切分方法（第 27 页）

#### 方法5：根据语义内容切分（第 27 页）

### 2.3.3 几个常用的文档切分器的方法的调用（第 28 页）

### 2.3.4 具体实现（第 28 页）

#### ① CharacterTextSplitter：Split by character（第 28 页）

##### 举例1：字符串文本的分割（第 28 页）

##### 举例2：指定分割符（第 29 页）

##### 举例3：指定分割符（第 30 页）

#### ② RecursiveCharacterTextSplitter：最常用（第 30 页）

##### 举例1：使用split_text()方法（第 31 页）

##### 举例2：使用create_documents()方法1（第 32 页）

##### 举例3：使用create_documents()方法2（第 34 页）

##### 举例4：使用split_documents()方法（第 41 页）

##### 举例5：自定义分隔符（第 43 页）

#### ③ TokenTextSplitter/CharacterTextSplitter：Split by tokens（第 44 页）

##### 举例1：使用TokenTextSplitter（第 45 页）

##### 举例2：使用CharacterTextSplitter（第 46 页）

#### ④ SemanticChunker：语义分块（第 47 页）

##### 1. breakpoint_threshold_type （断点阈值类型）（第 48 页）

##### 2. breakpoint_threshold_amount （断点阈值量）（第 49 页）

##### 3. sentence_split_regex （句子切分的正则表达式）（第 49 页）

#### ⑤ HTMLHeaderTextSplitter(了解)（第 49 页）

#### ⑥ CodeTextSplitter(了解)（第 50 页）

##### 举例1：支持的语言（第 51 页）

##### 举例2：python语言（第 51 页）

#### ⑦ MarkdownTextSplitter(了解)（第 51 页）

##### 举例：md数据类型（第 51 页）

## 2.4 文档嵌入模型 Text Embedding Models（第 52 页）

### 2.4.1 嵌入模型概述（第 52 页）

### 2.4.2 嵌入模型选型与初始化（第 54 页）

#### 选型：使用硅基流动平台的嵌入模型（第 54 页）

#### 初始化方式（第 54 页）

##### 初始化方式1：init_embeddings()（第 54 页）

##### 初始化方式2：OpenAIEmbeddings()（第 54 页）

### 2.4.3 句子的向量化（embed_query）（第 54 页）

### 2.4.4 文档的向量化（embed_documents）（第 55 页）

## 2.5 向量存储(Vector Stores)（第 56 页）

### 2.5.1 向量数据库的理解（第 56 页）

### 2.5.2 常用的向量数据库（第 57 页）

### 2.5.3 案例：民法典客服知识库（第 57 页）

#### ① 全局配置（第 58 页）

#### ② 初始化Milvus（第 58 页）

#### ③ 初始化 Embedding 模型（第 59 页）

#### ④ 读取文档并切分（第 60 页）

#### ⑤ 生成向量并写入 Milvus（第 62 页）

#### ⑥ 初始化模型与Agent（第 64 页）

#### ⑦ 检索逻辑 (Retrieval)（第 64 页）

#### ⑧ 生产与回答生成（第 65 页）
