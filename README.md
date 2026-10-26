# My First AI Agent (手搓 ReAct 架构)

## 📌 项目简介
本项目是从零开始，不依赖任何复杂框架（如 LangChain），纯手工实现的基于大语言模型（LLM）的 Agent 闭环。

## 🧠 核心架构 (TAOR 循环)
实现了标准的 Think -> Act -> Observe -> Repeat 循环：
1. **Think**: 将用户输入和工具列表发给 LLM。
2. **Act**: 解析 LLM 返回的 `tool_calls` JSON。
3. **Observe**: 在本地执行真实函数（壳），并将结果作为上下文发回。
4. **Repeat**: 直到 LLM 给出最终回答。

## 🛡️ 工程化兜底 (鲁棒性设计)
- **异常处理**：使用 `try-except` 捕获外部 API 报错，并转换为 `Observation` 喂回给模型，触发模型自我纠错。
- **死循环拦截**：设置了最大循环轮次（Max Turns = 5），防止模型卡死和 Token 无限消耗。

## 🚀 如何运行
1. 安装依赖：`pip install -r requirements.txt`
2. 复制 `.env.example` 为 `.env`，填入你的 LLM_API_KEY 等配置。
3. 运行：`python my_first_agent.py`

## 📸 运行效果（示例）
*(在这里放一张你刚才运行终端里，模型经历报错、自我纠正并最终降级输出的截图)*
