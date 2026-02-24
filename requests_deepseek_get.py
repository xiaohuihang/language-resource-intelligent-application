import requests
api_key = "sk-28b78b52e9ae42378cc766638b79c47f" #本地不需要
#url = "https://api.deepseek.com/models"
url = "http://192.168.43.44:11434/api/tags" #本地模型

headers = {
  "Accept": "application/json", #本地模型
  #"Authorization": f"Bearer {api_key}"
}
response = requests.get(url, headers=headers)
result = response.json()
#print(result['data']) #在线模型
print(result['models']) #本地模型