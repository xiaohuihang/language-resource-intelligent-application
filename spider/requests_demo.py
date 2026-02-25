import requests
# 发送一个GET请求到百度
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}
response = requests.get('http://192.168.43.227:5000/news',headers=headers)
# 打印返回内容
print(f'状态码: {response.status_code}')
print(f'网页内容预览: {response.text[:500]}')
print(f"状态码: {response.status_code}")
print(f"编码方式: {response.encoding}")
print(f"响应头类型: {response.headers['Content-Type']}")
print(f"请求头: {response.request.headers}")
print(f"URL: {response.url}")