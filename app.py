import numpy as np
import pickle as pkl
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score
import lightgbm as lgb
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from textblob import TextBlob
from flask import Flask, request,render_template,redirect,url_for
from flask_cors import CORS
from sklearn.externals import joblib
import pickle
import flask
import urllib
import pandas as pd
import numpy as np
import gzip
import re
import os
from scipy.sparse import hstack
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score
import lightgbm as lgb
from textblob import TextBlob
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from flask_restful import reqparse, abort, Api, Resource
from flask import Flask, flash, request, redirect, url_for
from flask import Flask, url_for, send_from_directory, request
import gunicorn
import glob
import os
import glob
import pickle
import pandas as pd
import chardet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.nearest_centroid import NearestCentroid       
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup as bs 
     
app = Flask(__name__)
CORS(app)

app=flask.Flask(__name__,template_folder='templates')

print("Loading models")
pickle_model = "models/wb_transform.pkl"
clf1= pkl.load(gzip.open(pickle_model, 'rb'))

class topic_classifier():
    """
    ***parameter***
    - file_dir: File path of BBC training data
    - topics: tweet topic from one of these topics: 'business','entertainment','politics','sport' and 'tech'
    - classifier: Machine learning multi-class classifier from one of the following classifiers
         +'mulNB': Naive Bayes 
         +'svc': SVC
         +'dec_tree': Decision Tree
         +'rand_forest': Random Forest
         +'random_sample': Random Sample
         +'nearest_cent': Nearest centroid
         +'mlp': Multi-layer Perceptron
    """
    def __init__(self,file_dir,topics,classifier):
        self.file_dir=file_dir
        self.topics=topics
        self.classifier=None
        self.algorithm=classifier
        self.method=None



    #extracting training texts from different folders and dumping them into a dataframe
    def train_topics_gen(self):
        content=[]
        classes=[]
        for topic in self.topics:
            user_set_path = os.path.join(self.file_dir,topic)
            os.chdir(user_set_path)
            files=glob.glob("*.txt")
            for file in files:
                with open(file) as f:
                    content.append(f.read())
                    classes.append(topic)
        DF = pd.DataFrame({'class': classes,'content': content})
        return DF


    # training the classifier using the BBC training data
    def training(self):
        if self.algorithm=='mulNB':
            self.classifier = MultinomialNB()
        elif self.algorithm=='svc':
            self.classifier=OneVsRestClassifier(SVC())
        elif self.algorithm=='dec_tree':
            self.classifier=DecisionTreeClassifier()
        elif self.algorithm=='rand_forest':
            self.classifier=RandomForestClassifier()
        elif self.algorithm=='random_sample':
            self.classifier=RandomForestClassifier()
        elif self.algorithm=='nearest_cent':
            self.classifier=NearestCentroid()
        elif self.algorithm=='mlp':
            self.classifier=MLPClassifier()

        # BBC training dataset
        df=self.train_topics_gen()
        # vectorizing the contents of the data
        self.method = CountVectorizer()
        counts = self.method.fit_transform(df['content'].values)
        targets = df['class'].values
        self.classifier.fit(counts, targets)
        return self

print("Loading classifier model")
pickle_model1 = "finalized_model.sav"
loaded_model = pickle.load(open(pickle_model1, 'rb'))

@app.route('/')
def main():
    return render_template('main.html')

non_alphanums = re.compile(u'[^A-Za-z0-9]+')
def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])

stemmer = SnowballStemmer("english")
def preprocess(df):
    df['author'].fillna('No author', inplace=True)
    df['title'].fillna('No title', inplace=True)
    df['text'].fillna('No text', inplace=True)
    df_author = pd.read_csv('author_cat.csv')
    df['author_cat'] = 1
    df['stemmed_title'] = df['title'].map(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))
    df['stemmed_text'] = df['text'].map(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))
    df.drop(['title', 'author', 'text'], axis=1, inplace=True)
    return df

@app.route('/predict',methods=['GET','POST'])
def predict():
    Author = request.form['Author']
    Title = request.form['Title']
    text = request.form['Text']
    d = {'title': [Title],
     'text': [text],
    'author': [Author]}
    df_test = pd.DataFrame(data=d)
    df= preprocess(df_test)
    vectorizer = HashingVectorizer(normalize_text,decode_error='ignore',n_features=2 ** 23, non_negative=False, ngram_range=(1, 2), norm='l2')
    X_title = vectorizer.transform(df['stemmed_title'])
    X_text = vectorizer.transform(df['stemmed_text'])
    X_author = df['author_cat'].values
    X_author = X_author.reshape(-1, 1)
    sparse_merge = hstack((X_title, X_text, X_author)).tocsr()
    mask100 = np.array(np.clip(sparse_merge.getnnz(axis=0) - 100, 1, 0), dtype=bool)
    X = sparse_merge[:, mask100]
    y1 = clf1.predict(X)
    ######sentimental analysis##############
    bloblist_desc = list()

    df_usa_descr_str=df_test['stemmed_text'].astype(str)
    for row in df_usa_descr_str:
        blob = TextBlob(row)
        bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
        df_usa_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['sentence','sentiment','polarity'])
    
    tweet_counts = loaded_model.method.transform(df_test['stemmed_text'])
    predictions = loaded_model.classifier.predict(tweet_counts)

    def f(df_usa_polarity_desc):
        if df_usa_polarity_desc['sentiment'] > 0:
           val = "Positive"
        elif df_usa_polarity_desc['sentiment'] == 0:
           val = "Neutral"
        else:
           val = "Negative"
        return val
    df_usa_polarity_desc['Sentiment_Type'] = df_usa_polarity_desc.apply(f, axis=1)
    df_usa_polarity_desc["Sentiment_Type"] = df_usa_polarity_desc.apply(func=f, axis=1)
    cal = np.round(y1, 5)*100
    if cal > 95 :
        m = "This News is Fake"
    elif cal > 80 and cal < 95:
        m = "This News is more likely a Fake"
    else:
        m = "This News is Real"
    return render_template('main.html', prediction_text= "Fake Rate={}".format(np.round(y1, 4)*100)+"%"+"->"+m,para = "Sentiments={}".format(df_usa_polarity_desc["Sentiment_Type"].values),para1="Category={}".format(predictions))

@app.route('/predict1',methods=['GET','POST'])
def predict1():
    video_url = request.form['video_id']
    match = re.search(r"youtube\.com/.*v=([^&]*)", video_url)
    if match:
       result = match.group(1)
    else:
       result = ""
    # get the html content
    content = requests.get(video_url)
    soup = bs(content.content, "html.parser")
    title = soup.find("span", attrs={"class": "watch-title"}).text.strip()
    name =  soup.find("div", attrs={'class': "yt-user-info"}).text.strip()
    try:
        text = YouTubeTranscriptApi.get_transcript(result)
        text_new = []
        for i in range(0,len(text)):
           text_new.append(text[i]['text'])
           text_new1 = ' '.join(text_new)
    except:
        text = soup.find("span", attrs={"class": "watch-title"}).text.strip()
        text_new1 = text
    
    text_new1 = text_new1.replace('\n', '')
    text_new1 = text_new1.replace('\'', '')
    text_new1 = text_new1.replace('"', '')
    d = {'title': [title],
     'text': [text_new1],
    'author': [name]}
    df_test = pd.DataFrame(data=d)
    df= preprocess(df_test)
    vectorizer = HashingVectorizer(normalize_text,decode_error='ignore',n_features=2 ** 23, non_negative=False, ngram_range=(1, 2), norm='l2')
    X_title = vectorizer.transform(df['stemmed_title'])
    X_text = vectorizer.transform(df['stemmed_text'])
    X_author = df['author_cat'].values
    X_author = X_author.reshape(-1, 1)
    sparse_merge = hstack((X_title, X_text, X_author)).tocsr()
    mask100 = np.array(np.clip(sparse_merge.getnnz(axis=0) - 100, 1, 0), dtype=bool)
    X = sparse_merge[:, mask100]
    y1 = clf1.predict(X)
    ######sentimental analysis##############
    bloblist_desc = list()

    df_usa_descr_str=df_test['stemmed_text'].astype(str)
    for row in df_usa_descr_str:
        blob = TextBlob(row)
        bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
        df_usa_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['sentence','sentiment','polarity'])
    
    tweet_counts = loaded_model.method.transform(df_test['stemmed_text'])
    predictions = loaded_model.classifier.predict(tweet_counts)

    def f(df_usa_polarity_desc):
        if df_usa_polarity_desc['sentiment'] > 0:
           val = "Positive"
        elif df_usa_polarity_desc['sentiment'] == 0:
           val = "Neutral"
        else:
           val = "Negative"
        return val
    df_usa_polarity_desc['Sentiment_Type'] = df_usa_polarity_desc.apply(f, axis=1)
    df_usa_polarity_desc["Sentiment_Type"] = df_usa_polarity_desc.apply(func=f, axis=1)
    cal = np.round(y1, 5)*100
    if cal > 95 :
        m = "This News is Fake"
    elif cal > 80 and cal < 95:
        m = "This News is more likely a Fake"
    else:
        m = "This News is Real"

    #return render_template('main.html', prediction_text= "Fake Rate={}".format(np.round(y1, 4)*100)+"%"+"--->"+m+ "   Sentiments--->"+df_usa_polarity_desc["Sentiment_Type"].values+"Category--->"+predictions)
    return render_template('main.html', prediction_text= "Fake Rate={}".format(np.round(y1, 4)*100)+"%"+"->"+m,para = "Sentiments={}".format(df_usa_polarity_desc["Sentiment_Type"].values),para1="Category={}".format(predictions))


    
if __name__=="__main__":
     #port=int(os.environ.get('PORT',5000))
     app.run()

    
