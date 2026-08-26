# 05_prompttemplates 大纲

# 1.提示词模板(Prompt Templates)（第 1 页）

# 1.1 复习：str.format()（第 1 页）

## 1）带有位置参数的用法（第 2 页）

## 2）带有关键字参数的用法（第 2 页）

## 3）使用字典解包的方式（第 2 页）

## 4）字符串拼接方式（第 2 页）

# 1.2 提示词模板（第 3 页）

## 1）PromptTempalte --langchain1.0之后弱化了（第 3 页）

### （1）两种实例化方式（第 4 页）

#### 方式1：构造方法 format()（第 4 页）

##### 举例1：单变量（第 4 页）

##### 举例2：定义多变量（第 4 页）

#### 方式2：调用from_template() 推荐！！！（第 4 页）

### （2）两种新的结构形式（第 5 页）

#### 形式1：部分提示词模版（第 5 页）

##### 方式1：实例化过程中使用partial_variables变量（第 5 页）

##### 方式2：使用 PromptTemplate.partial() 方法创建部分提示模板（第 5 页）

#### 形式2：组合提示词(了解)（第 6 页）

### （3）给变量赋值的两种方式：format() 与 invoke()（第 7 页）

### （4）结合LLM调用（第 8 页）

## 2）ChatPromptTemplate（第 8 页）

### （1）两种实例化方式（第 9 页）

#### 方式1：使用实例初始化方法（第 9 页）

#### 方式2：调用from_messages()--推荐（第 10 页）

### （2）调用提示词模板的几种方法（第 10 页）

##### 方法1：invoke() ---返回ChatPromptValue（第 10 页）

##### 方法2：format() ---返回str（第 11 页）

##### 方法3：format_messages() ---返回消息构成的list（第 11 页）

##### 方法4：format_prompt() ---返回ChatPromptValue（第 11 页）

##### 如何实现ChatPromptValue与list[messages]、字符串之间转换？（第 12 页）

### （3）更丰富的实例化参数类型（第 13 页）

##### 举例1：元组列表（第 13 页）

##### 举例2：字符串列表（第 14 页）

##### 举例3：字典列表（第 14 页）

##### 举例4：消息对象列表（第 15 页）

##### 举例5：BaseChatPromptTemplate参数列表（第 16 页）

##### 举例6：BaseMessagePromptTemplate参数列表（第 16 页）

##### 综合案例（第 17 页）

### （4）结合LLM（第 18 页）

### （5） ChatPromptTemplate的高级特性（第 19 页）

#### 1）部分变量的预填充： partial()（第 19 页）

#### 2）消息占位符（第 20 页）

##### JSON形式（第 20 页）

##### 插入消息列表：MessagesPlaceholder（第 21 页）

#### 3）存储对话历史内容（第 22 页）

#### 4）可复用的模板库（第 23 页）

##### 举例1：template.py文件声明模板库（第 23 页）

##### 举例2：分类创建模板（第 24 页）

#### 5）模板组合（第 24 页）

##### 方法1：字符串组合（第 24 页）

##### 方法2：使用+运算符（第 25 页）

# 1.3 少量样本示例的提示词模板（第 25 页）

## 1）使用说明（第 25 页）

## 2）FewShotPromptTemplate使用（第 25 页）

### 举例1 未提供示例的情况（第 25 页）

### 举例2 使用FewShotPromptTemplate（第 26 页）

## 3）FewshotChatMessagePromptTemplate的使用（第 27 页）

## 4）Example selectors(示例选择器)（第 29 页）

### 举例1：SemanticSimilarityExampleSelector（第 29 页）

### 举例2：结合FewShotPromptTemplate使用（第 31 页）

# 5）PipelinePromptTemplate(了解)（第 32 页）

# 6）自定义提示词模版(了解)（第 34 页）

# 7）从文档中加载Prompt(了解)（第 35 页）

## （1）yaml格式提示词（第 35 页）

## （2）json格式提示词（第 35 页）

## （3）调用py文件提示词模板（第 36 页）
