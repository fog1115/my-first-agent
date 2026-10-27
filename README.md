# AI Agent & RAG 学习实践项目

本项目记录了从零开始学习 AI Agent 开发的实战历程，包含两个核心模块：手搓 ReAct 架构，以及基于 Chroma 的 RAG（检索增强生成）链路。

## 📂 项目结构
- `my_first_agent.py`：手搓的 ReAct Agent 核心代码。
- `second-real-agent.py`：基础的 RAG（检索增强生成）测试代码。
- `requirements.txt`：项目依赖环境。

---

## 📌 模块一：手搓 ReAct 架构 Agent
### 🧠 核心实现
不依赖任何复杂框架（如 LangChain），纯手工实现基于大语言模型的 Agent 闭环。
*   **TAOR 循环**：实现了标准的 思考(Think) -> 行动(Act) -> 观察(Observe) -> 循环(Repeat)。
*   **Function Calling**：解析 LLM 返回的 `tool_calls`，在本地执行真实函数，并将结果作为上下文发回。

### 🛡️ 工程化兜底
*   **异常处理**：使用 `try-except` 捕获工具报错，转换为 Observation 喂回给模型，触发模型自我纠错。
*   **死循环拦截**：设置最大循环轮次（Max Turns = 5），防止模型卡死和 Token 无限消耗。

---

## 📌 模块二：RAG 知识库问答系统
### 🧠 核心实现
针对大模型“幻觉”问题，实现了基于本地向量数据库的检索增强生成。
*   **文档切分 (Chunking)**：使用 `RecursiveCharacterTextSplitter` 对长文本进行重叠切分。
*   **向量化 (Embedding)**：加载本地 `all-MiniLM-L6-v2` 模型，将文本转化为数学向量。
*   **向量检索**：使用 `ChromaDB` 存储向量，利用余弦相似度召回 Top-K 相关片段。
*   **防幻觉约束**：利用 Prompt 严格限制大模型，若资料中无答案，必须明确拒绝回答。

---

## 🚀 如何运行
1. 安装依赖：`pip install -r requirements.txt`
2. 复制 `.env.example` 为 `.env`，填入你的 `LLM_API_KEY` 等配置。
3. 运行 Agent：`python my_first_agent.py`
4. 运行 RAG：`python second-real-agent.py`

## 📸 运行效果
*(此处可以放一张你刚才跑通 RAG 的终端截图，用图片展示你的成果)*

