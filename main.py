# Imports

from flask_restful import Api, Resource
from flask import Flask
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from nltk import word_tokenize
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import nltk
import sklearn
import warnings
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)
api = Api(app)


class Categorization(object):

    def __init__(self):
        # Data Collection

        df = pd.read_csv("train.csv")
        df2 = pd.read_csv("test.csv")

        # Take relevant data from DataFrame and put it in Numpy Array

        tweets = df['text'].tolist()
        labels = df['class_label'].to_numpy()
        testTweets = df2['text'].tolist()
        testLabels = df2['class_label'].to_numpy()

        # TFID Vectorizer

        def preprocess(s):
            lemmatizer = nltk.WordNetLemmatizer()
            return lemmatizer.lemmatize(s)

        stop = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(stop_words=stop, analyzer='word',
                                          max_features=20000, dtype=np.float32, preprocessor=preprocess)
        data = self.vectorizer.fit_transform(tweets).toarray()
        testData = self.vectorizer.transform(testTweets).toarray()

        #X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.3, random_state=10)
        X_train = data
        Y_train = labels
        X_test = testData
        Y_test = testLabels

        self.clf = BernoulliNB()
        self.clf.fit(X_train, Y_train)
        BernoulliNB(alpha=2.0, binarize=0.0, class_prior=None, fit_prior=True)

        predictions = self.clf.predict(X_test)

    """
    # Stopwords part

    def not_stopword(s):
    s = s.strip()
    v = stopwords.words('english')
    result = ""
    words = nltk.word_tokenize(s)
    for word in words:
        if word not in v:
        result += word + " "
    return result.strip()

    i=0
    for token in tokens:
    token = preprocess(token)
    
    finalsentence = ' '.join(tweet.split())


    print(finalsentence)
    """

    def processData(self, inputString):
        liveData = self.vectorizer.transform([inputString]).toarray()

        # print(liveData)

        ans = self.clf.predict(liveData).tolist()
        return ans[0]


class HelloWorld(Resource):
    def __init__(self):
        self.categorization = Categorization()

    def get(self, inputString):
        return {"prediction": self.categorization.processData(inputString)}


api.add_resource(HelloWorld, "/helloworld/<string:inputString>/")

if __name__ == "__main__":
    app.run(debug=True)
