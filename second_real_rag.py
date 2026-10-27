import os
# 强制开启离线模式，告诉程序：别去外网查了，用我本地下载好的！
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from openai import OpenAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

# 初始化大模型客户端（DeepSeek）
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

# ==========================================
# 1. 准备数据
# ==========================================
long_document = """
关于2024年计算机学院保研政策的详细说明。
第一条：申请保研的学生，必须在前六个学期无任何挂科记录。如果有挂科记录，则直接取消保研资格。
第二条：保研的综合成绩由绩点（占80%）和科研竞赛加分（占20%）组成。绩点必须达到3.5以上。
第三条：关于科研竞赛加分，获得国家级一等奖加10分，省级一等奖加5分。
第四条：保研名额有限，计算机科学与技术专业今年共有20个名额，软件工程专业有15个名额。
第五条：面试环节占总成绩的30%，主要考察算法基础和项目经验。如果面试不及格，一票否决。
"""

# ==========================================
# 2. 文本切分
# ==========================================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = text_splitter.split_text(long_document)
print(f"--- 文档被切分成了 {len(chunks)} 个片段 ---")

# ==========================================
# 3. 加载本地模型并手动计算向量
# ==========================================
print("--- 正在加载本地模型并计算向量 ---")
# 【注意】这里指向你刚才下载的那个文件夹。如果你代码和文件夹不在同一个目录，请改成绝对路径。
model = SentenceTransformer('C:/Users/yyyy1/all-MiniLM-L6-v2') 

# 把切分好的文档变成向量
embeddings = model.encode(chunks).tolist()
print("向量计算完成！")

print("--- 正在存入向量数据库 ---")
chroma_client = chromadb.Client()
# 这里我们不要它自己去算向量了，所以不传 embedding_function
collection = chroma_client.create_collection(name="baoyan_policy")

ids = [f"chunk_{i}" for i in range(len(chunks))]
collection.add(
    documents=chunks,
    embeddings=embeddings, # 把刚才算好的向量传给数据库
    ids=ids
)
print("入库成功！")

# ==========================================
# 4. 检索与生成
# ==========================================
def ask_rag(question):
    print(f"\n用户提问: {question}")
    
    # 把用户的问题也变成向量
    query_embedding = model.encode([question]).tolist()
    
    # 去数据库里搜最相似的片段
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2 
    )
    
    retrieved_chunks = results['documents'][0]
    print(f"--- 向量库检索到以下资料 ---")
    for idx, chunk in enumerate(retrieved_chunks):
        print(f"[{idx+1}] {chunk}")
        
    context = "\n".join(retrieved_chunks)
    
    # 构造 Prompt
    prompt = f"""
    你是一个专业的保研政策解答助手。请严格根据下面提供的政策资料来回答用户的问题。
    如果资料中找不到答案，请直接说“根据现有资料，我无法回答”，绝对不要编造。
    
    【政策资料】：
    {context}
    
    【用户问题】：
    {question}
    """
    
    # 请求大模型
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL_ID", "deepseek-chat"),
        messages=[{"role": "user", "content": prompt}]
    )
    
    print(f"\n--- 大模型最终回答 ---")
    print(response.choices[0].message.content)

# ==========================================
# 5. 测试
# ==========================================
if __name__ == "__main__":
    # 测试1：问一个资料里有的
    ask_rag("挂科了还能保研吗？")
    
    # 测试2：问一个资料里没有的（测试幻觉）
    ask_rag("保研需要考英语听力吗？")
