import os 
import time 
import datetime
import threading 
import multiprocessing as mp 

import pandas as pd 

from selenium import webdriver 
from selenium.webdriver.common.keys import Keys 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC 
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager



start_date = "01-01-2015"
end_date = "01-09-2022"

with open("checked_dates.txt", "r") as f:
    checked_dates = [l.strip("\n") for l in f]

def get_dates():
    start = datetime.datetime.strptime(start_date, "%d-%m-%Y")
    end = datetime.datetime.strptime(end_date, "%d-%m-%Y")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
    dates = [date.strftime("%d-%m-%Y") for date in date_generated]
    return dates

class StockNews(threading.Thread):
    def __init__(self, date, all_news_link):
        threading.Thread.__init__(self)
        self.all_news_link = all_news_link
        self.date = date 
        self.df_path = "news_titles.csv"
        if not os.path.exists(self.df_path):
            df = pd.DataFrame(columns=["date", "title"])
            df.to_csv(self.df_path, index=False)

    def get_stock_news(self):
        if self.all_news_link is not None and len(self.all_news_link) > 0:
            try:
                stock_news = [(link, link.rsplit("/", 1)[1].replace("-", " ")) for link in self.all_news_link 
                            if any (x in link for x in ["/stock/", "/stock-corporate/"])]
                # stock_news = [news for news in stock_news if "date" not in news]
                print(stock_news)
                return stock_news 
            except Exception as e:
                print(str(e))
                return []

    def write_csv(self):
        stock_news = self.get_stock_news()
        if len(stock_news) > 0:
            data = {
                "date": [self.date] * len(stock_news),
                "title": stock_news
            }
            df = pd.DataFrame(data)
            df.to_csv(self.df_path, mode="a", header=False, index=False)

    def run(self):
        self.write_csv()

options = Options()
options.add_argument('--headless')

def get_page_data(date):
    if date not in checked_dates:
        try:
            driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=options)
            link = "https://today.thefinancialexpress.com.bd/public/?date={}".format(date)
            driver.get(link)
            time.sleep(30)
            posts = driver.find_elements(
                By.TAG_NAME, "a"
            )
            all_news_link = [post.get_attribute("href") for post in posts]
            all_news_link = [x for x in all_news_link if x is not None]
            with open("checked_dates.txt", "a") as f:
                f.write(date + "\n")
            obj = StockNews(date, all_news_link).start()
            obj.join()
            driver.close()
            
        except Exception as e:
            with open("failed_dates.txt", "a") as f:
                f.write(date + "\n")
            print(str(e))



if __name__ == "__main__":
    dates = get_dates()
    mp.Pool(5).map(get_page_data, dates)
  
