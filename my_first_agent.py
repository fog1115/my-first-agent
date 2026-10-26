import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 1. 加载环境变量（会自动读取你刚才建好的 .env 文件）
load_dotenv()

# 2. 初始化大模型客户端（以 DeepSeek 为例，兼容 OpenAI 格式）
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

# 3. 定义工具（这就是壳里的“手脚”）
def get_weather(city):
    if city == "杭州":
        return f"{city}今天天气晴朗，气温20度。"
    else:
        raise ValueError(f"天气系统故障：找不到 {city} 的天气数据！")

def get_attraction(city, weather):
    # 假装我们调用了景点推荐API
    return f"结合{city}的{weather}，推荐去西湖边散步。"
def get_calculate_budget(days):
    t=days*500
    return f"旅游总预算为 {t} 元。"

# 告诉模型我们有哪些工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_attraction",
            "description": "根据天气获取推荐景点",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "weather": {"type": "string"}
                },
                "required": ["city", "weather"]
            }
        }
    },
        {
        "type": "function",
        "function": {
            "name": "get_calculate_budget",
            "description": "获取旅游预算,每天花费 500 元",
            "parameters": {
                "type": "object",
                "properties": {"days": {"type":"integer"}},
                "required": ["days"]
            }
        }
    },
]

# 4. 真正的 Agent 循环 (TAOR: Think -> Act -> Observe -> Repeat)
def run_agent(user_query):
    messages = [{"role": "user", "content": user_query}]
    
    print(f"\n用户提问: {user_query}\n")
    print("--- 开始思考 ---")

    # 限制最多循环 5 次，防止死循环
    for turn in range(5):
        # [Think]：把对话和工具列表发给模型，让它做决策
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL_ID", "deepseek-chat"),
            messages=messages,
            tools=tools
        )
        
        response_message = response.choices[0].message
        messages.append(response_message) # 把模型的回复加入历史
        
        # [判断]：模型是不是觉得可以直接回答了？
        if not response_message.tool_calls:
            print(f"\n最终回答: {response_message.content}")
            break
            
        # [Act]：模型决定调用工具了！
        for tool_call in response_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            print(f"行动 (Action): 调用工具 {tool_name}，参数: {tool_args}")
            
            # [执行]：壳在真正执行代码
            try:
                if tool_name == "get_weather":
                    result = get_weather(**tool_args)
                elif tool_name == "get_attraction":
                    result = get_attraction(**tool_args)
                elif tool_name == "get_calculate_budget":
                    result = get_calculate_budget(**tool_args)
                else:
                    result = "工具不存在"
            except Exception as e:
                # 把错误信息转化为字符串，告诉大模型
                result = f"工具调用失败，报错信息: {str(e)}"
                result = f"工具调用失败，报错信息: {str(e)}"
                
            print(f"观察 (Observation): {result}")
            
            # [Observe]：把执行结果作为观察，塞回给模型
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": result
            })
        print("\n--- 继续思考 ---")

# 5. 启动！
if __name__ == "__main__":
    run_agent("帮我查一下纽约现在的天气，并根据天气推荐一个景点,并顺便算一下如果玩3天需要多少预算。")
