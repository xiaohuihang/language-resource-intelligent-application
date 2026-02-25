import requests
from bs4 import BeautifulSoup
import csv

def crawl_news_site():
    # 获取页面
    url = "http://127.0.0.1:5000/news"
    response = requests.get(url)

    # 解析HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # 提取新闻数据
    news_items = soup.find_all('article', class_='news-item')

    data = []
    for item in news_items:
        news_data = {
            'id': item.get('data-id'),
            'title': item.find('h3', class_='news-title').text,
            'author': item.find('span', class_='author').text,
            'date': item.find('span', class_='date').text,
            'category': item.get('data-category'),
            'image_url': item.find('img', class_='news-image')['src'],
            'image_alt': item.find('img', class_='news-image')['alt']
        }
        data.append(news_data)

    # 保存为CSV
    with open('../news_data.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    print(f"共爬取 {len(data)} 条新闻数据")
    return data

if __name__ == "__main__":
    crawl_news_site()