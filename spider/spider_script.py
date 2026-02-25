import requests
from bs4 import BeautifulSoup

# 1. 获取页面内容
url = "http://192.168.43.227:5000/news"
response = requests.get(url)

# 2. 解析HTML
soup = BeautifulSoup(response.text, 'html.parser')

# 3. 提取所有新闻标题
titles = soup.find_all('h3', class_='news-title')
for i, title in enumerate(titles, 1):
    print(f"新闻{i}: {title.text}")