import joblib
import pandas as pd
from flask import Flask, request
from text_filter import my_review_filter
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from amazon.amazon.spiders.amspy import AmspySpider
from multiprocessing import Process

model = None
vectorizer = None

def init():
    global model,vectorizer
    model = joblib.load(r'./Model/trained_model.pkl')
    vectorizer = joblib.load(r'./Model/trained_vectorizer.pkl')

def startCrawling(link):
    process = CrawlerProcess(get_project_settings())
    process.crawl(AmspySpider, slink=link)
    process.start()
    process.join()

def runSpider(link):
    try:
        crawlProcess = Process(target=startCrawling, args=(link,))
        crawlProcess.start()
        crawlProcess.join() 
        print("SCRAPY RUNNED")
        return True
    except Exception as e:
        print(f"Error in running scrapy ${e}")
        return False

def sentimentSuccess():
    try:
        df1 = pd.read_csv(r'./data.csv')
        X = vectorizer.transform(df1['Review'])
        sentiment = model.predict(X)
        probability = model.predict_proba(X)
        sentiment=sentiment.flatten().tolist()
        confidence=[]
        for i in range(len(sentiment)):
            if sentiment[i] == 1:
                sentiment[i] = 'Positive'
                confidence.append(round(100*probability[i][1],2))
            else:
                sentiment[i] = 'Negative'
                confidence.append(round(100*probability[i][0],2))
        df2=pd.DataFrame({'Sentiment':sentiment,'Confidence':confidence})
        df1['Sentiment']=df2['Sentiment']
        df1['Confidence']=df2['Confidence']
        df1.to_csv('./data.csv', index=False)
        return True
    except Exception as e:
        print(f"An Error occurred at parsing data of csv. : {e}")
        return False


# Flask
app = Flask(__name__)

@app.route('/get-amazon-reviews', methods=['POST'])
def result():
    if(model==None):init()
    try:
        link = request.form['link']
        # empty the data.csv
        df=pd.DataFrame()
        df.to_csv('./data.csv', index=False)
        # start scrapy and get result
        if runSpider(link) and sentimentSuccess():
            df = pd.read_csv(r'./data.csv')
            output = df.to_json(orient='records')
            return output, 200
        else:
            return "Error in processing.", 500
    except Exception as e:
        print(f"Error in processing: {e}")
        return "Error in processing.", 500

if __name__ == '__main__':
    app.run()
