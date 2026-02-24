import requests
import json
# 1. 配置Ollama服务地址和模型名称
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "nomic-embed-text"  # 替换为你启动的模型名（如bge、llama3-embed等）

# 2. 定义要生成嵌入的文本
text = "不支持全屏播放；摄像头没有提供闪光灯的支持；TF存储卡不支持热插拔；不支持蓝牙传输功能。"
# 3. 构造请求体
payload = {
    "model": MODEL_NAME,  # 必须：指定使用的模型名
    "prompt": text,       # 必须：要生成嵌入的文本
    # 可选：自定义模型参数（如温度、维度，根据模型支持情况调整）
    "options": {
        "temperature": 0.0  # embedding模型建议设为0，保证结果稳定
    }
}
# 4. 发送POST请求调用Ollama API
try:
    response = requests.post(
        OLLAMA_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    # 5. 解析响应
    if response.status_code == 200:
        result = response.json()
        print(result)
        # 提取核心的embedding向量（一维数组，长度随模型不同，如Qwen3是4096维）
        embedding = result["embedding"]
        print(f"✅ 生成的Embedding向量长度：{len(embedding)}")
        print(f"🔍 向量前10个值：{embedding[:10]}")
    else:
        print(f"❌ 请求失败，状态码：{response.status_code}，错误信息：{response.text}")

except Exception as e:
    print(f"❌ 调用出错：{str(e)}")