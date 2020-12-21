import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import random
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# Question 2: Cleaning the data

def remove_html_tags(raw_text):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_text)
    return cleantext

def preprocess(raw_text):

    # Removing punctuation and numbers
    raw_text = re.sub(r'[^a-z\s]', ' ', raw_text)

    # Tokenize raw text
    word_list = nltk.word_tokenize(raw_text)

    # Init the Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize list of words and join
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    lemmatized_output = [w for w in lemmatized_output.split() if not w in stop_words]

    return " ".join(lemmatized_output)

def tfidf_svd(data):
    vectorizer = TfidfVectorizer(min_df=0.1)
    tfidf_data = vectorizer.fit_transform(data)
    
    temp_components = tfidf_data.shape[1] - 1 if tfidf_data.shape[1] <= 100 else 100
    svd = TruncatedSVD(n_components=temp_components, random_state=42)
    svd.fit(tfidf_data)
    
    variance = np.cumsum(svd.explained_variance_ratio_)
    n_components = np.argmax(variance >= 0.6) + 1
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    data_reduced = svd.fit_transform(tfidf_data)
    
    return data_reduced