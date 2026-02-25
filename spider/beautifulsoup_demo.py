
import requests
from bs4 import BeautifulSoup
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}
response = requests.get('http://192.168.43.227:5000/news',headers=headers)
# 打印返回内容
html_content = response.text  # 这是完整的、杂乱的HTML字符串
soup = BeautifulSoup(html_content, 'html.parser')

# 1. 找到页面主标题
main_title = soup.find('h1')
print(f"页面主标题: {main_title.text}")

# 2. 找到所有新闻项（根据你的网站结构调整选择器）
news_items = soup.find_all('article', class_='news-item')  # 或 article.news-item
print(f"找到 {len(news_items)} 条新闻")
#提取文本内容
for item in news_items:
    title = item.find('h3', class_='news-title').text  # 在item内继续查找
    content = item.find('p', class_='content').text
    print(f"标题: {title}")
    print(f"内容: {content}")
#提取图像内容
for item in news_items:
    image_url = item.find('img', class_='news-image')['src'],
    image_alt = item.find('img', class_='news-image')['alt']
    print(f"image:{image_url}")
    print(f"alt:{image_alt}")

# 3. 找到第一张新闻图片
first_image = soup.find('img')
print(f"第一张图片描述: {first_image.get('alt', '无描述')}")