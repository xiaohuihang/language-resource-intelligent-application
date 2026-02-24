import requests
import json

# 设置 API 密钥和端点
api_key = "sk-28b78b52e9ae42378cc766638b79c47f" #本地部署不需要
#url = "https://api.deepseek.com/v1/chat/completions"
url = "http://192.168.43.227:11434/api/chat" #本地模型使用

# 准备请求头
headers = {
    "Content-Type": "application/json",
    #"Authorization": f"Bearer {api_key}" #在线模型使用
}
"""
# 准备请求数据——在线模型使用
data = {
    "model": "deepseek-chat",  # 或 "deepseek-coder"
    "messages": [
        {"role": "system", "content": "你是一个智能助理"},
        {"role": "user", "content": "你知道情感词库吗?"}
    ],
    "stream": False,
    "max_tokens": 1024
}

"""
#准备请求数据——本地模型使用
data = {
    "model": "deepseek-r1:1.5b",  # 或 "deepseek-coder"
    "messages": [
        {"role": "system", "content": "你是一个智能助理"},
        {"role": "user", "content": "你知道情术语库吗?"}
    ],
    "stream": False,
    "max_tokens": 1024
}

# 发送请求
response = requests.post(url, headers=headers, json=data)

# 处理响应
if response.status_code == 200:
    result = response.json()
    #print(result['choices'][0]['message']['content']) #在线模型使用
    print(result['message']['content']) #本地模型使用
else:
    print(f"Error: {response.status_code}")
    print(response.text)