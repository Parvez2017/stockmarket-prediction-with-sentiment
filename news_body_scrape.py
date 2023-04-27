import os 
import pandas as pd 
import ast 
import tqdm 
from threading import Thread
import multiprocessing as mp 

import requests
from bs4 import BeautifulSoup

class Saver(Thread):
    def __init__(self, date, title, link, body):
        Thread.__init__(self)
        self.date = date 
        self.title = title
        self.link = link 
        self.body = body 
        self.savepath = "news_body.tsv"
        if not os.path.exists(self.savepath):
            df = pd.DataFrame(columns=['date', 'link', 'title', 'body'])
            df.to_csv(self.savepath, sep="\t", index=False)

    def run(self):
        data = {
            "date": [self.date],
            "link": [self.link],
            "title": [self.title],
            "body": [self.body]
            }
        df = pd.DataFrame(data)
        df.to_csv(self.savepath, sep='\t', mode="a", header=False, index=False)
    
def main(row):
    try:
        
        date = row['date']
        # content = ast.literal_eval(row['title'])
        link = row['link']
        title = row['title']
        if not "?date=" in title:
            # print(type(row['title']))
            page = requests.get(link)
            soup = BeautifulSoup(page.content, 'html.parser')
            ls = [data.getText() for data in soup.find_all("p")]
            s = "".join(ls)
            saver = Saver(date, title, link, s)
            saver.start()
            saver.join()
    except Exception as e:
        print(str(e))


if __name__ == "__main__":
    df = pd.read_csv('new_news_titles.csv')
    rows = [row for _, row in df.iterrows()]
    mp.Pool(5).map(main, rows)