import os 
import tqdm 
import pandas as pd 
import torch 
from threading import Thread
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

class Saver(Thread):
    def __init__(self, date, title, link, body, company):
        Thread.__init__(self)
        self.date = date 
        self.title = title
        self.link = link 
        self.body = body 
        self.company = company
        self.ap = len(company)
        self.savepath = "company_wise_news_2.tsv"
        if not os.path.exists(self.savepath):
            df = pd.DataFrame(columns=['date', 'link', 'title', 'body', 'company'])
            df.to_csv(self.savepath, sep="\t", index=False)

    def run(self):
        data = {
            "date": [self.date] * self.ap,
            "link": [self.link] * self.ap,
            "title": [self.title] * self.ap,
            "body": [self.body] * self.ap,
            "company": self.company
            }
        df = pd.DataFrame(data)
        df.to_csv(self.savepath, sep='\t', mode="a", header=False, index=False)
    

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer,framework="pt", device=device)

    df = pd.read_csv("news_body_new.tsv", sep="\t")
    
    for _, row in tqdm.tqdm(df.iterrows()):
        try:
            print(row)
            date, link, title, body = row
            p = nlp(body)
            i = 0
            ls = []
            while(i < len(p)):
                if p[i]['entity'] == "B-ORG":
                    m = p[i]['word']
                    i = i + 1
                    if i >= len(p):
                        break
                    if p[i]['entity'] == "I-ORG":
                        while(p[i]['entity'] == "I-ORG"):
                            if p[i]['word'][0] == "#":
                                m += p[i]['word'].replace("#", '')
                                # i += 1
                            else:
                                m += " " + p[i]['word']
                    
                            i = i + 1
                            if i >= len(p):
                                break
                        print(m)
                        ls.append(m.title())
                else:
                    i = i + 1
                    if i >= len(p):
                        break

            saver = Saver(date, link, title, body, ls)
            saver.start()
            saver.join()
        except Exception as e:
            print(str(e))