import os 
import time 
import datetime
import pandas as pd 

from selenium import webdriver 
from selenium.webdriver.common.keys import Keys 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC 
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

driver = webdriver.Chrome(ChromeDriverManager().install())

start_date = "01-01-2015"
end_date = "01-09-2022"

def get_dates():
    start = datetime.datetime.strptime(start_date, "%d-%m-%Y")
    end = datetime.datetime.strptime(end_date, "%d-%m-%Y")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
    dates = [date.strftime("%d-%m-%Y") for date in date_generated]
    print(dates)
    return dates

def get_page_data(date):
    link = f"https://today.thefinancialexpress.com.bd/public/?date={date}"
    driver.get(link)
    time.sleep(30)
    posts = driver.find_elements(
        By.TAG_NAME, "a"
    )
    all_news_link = [post.get_attribute("href") for post in posts]
    all_news_link = [x for x in all_news_link if x is not None]
    driver.close()
    return all_news_link 

def get_stock_news(all_news_link):
    if not os.path.exists("news_title.csv"):
        df = pd.DataFrame()
        df.to_csv("news_title.csv")
    print(len(all_news_link))
    stock_news = [link.rsplit("/", 1) for link in all_news_link 
                  if "/stock/" or "/stock-corporate/" in link]
    print(stock_news)
    

news_links = get_page_data(end_date)
get_stock_news(news_links)