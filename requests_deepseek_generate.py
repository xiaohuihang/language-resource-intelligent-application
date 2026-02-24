import requests
import json

api_key = "sk-28b78b52e9ae42378cc766638b79c47f" #在线模型使用
#url = "https://api.deepseek.com/beta/completions" 在线模型使用
url = "http://192.168.43.44:11434/api/generate" #本地模型使用
headers = {
    "Content-Type": "application/json",
    #"Authorization": f"Bearer {api_key}" #在线模型使用
}
"""
# 准备请求数据——在线模型使用
data = {
  "model": "deepseek-chat",
  "prompt": "我是一名国防科技大学的研究生,我",
  "max_tokens": 1024,
  "stream": False
}
"""
# 准备请求数据 本地部署使用
data = {
    "model": "deepseek-r1:7b",  # 或你安装的其他模型，如  "qwen2.5" 等
    "prompt": "我是一名国防科技大学的研究生,我",
    "max_tokens": 1024,
    "stream": False
}

# 发送请求
response = requests.post(url, headers=headers,json=data)

# 处理响应
if response.status_code == 200:
    result = response.json()
    #print(result['choices'][0]['text'])  # 在线模型使用
    print(result["response"]) #本地模型使用

else:
    print(f"Error: {response.status_code}")
    print(response.text)