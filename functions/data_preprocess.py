import numpy as np
import pandas as pd
import pickle
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
nltk.download('averaged_perceptron_tagger')

# Question 2: Cleaning the data

def remove_html_tags(raw_text):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_text)
    return cleantext

def preprocess(raw_text):
    """Apply standard NLP preprocess (html tags removal, punctuation removal, std + custom stopwords removal)."""
    raw_text = raw_text.lower()
    raw_text = remove_html_tags(raw_text)
    # Removing punctuation and numbers
    raw_text = re.sub(r'[^a-z\s]', ' ', raw_text)
    # Tokenize raw text
    word_list = nltk.word_tokenize(raw_text)
    
    # Removing stopwords (standard stopwords + domanin specific stopwords)
    std_stopwords = set(stopwords.words('english'))
    with open('./functions/custom_stopwords.p', 'rb') as fp:
        custom_stopwords = pickle.load(fp)
    stop_words = set.union(std_stopwords, custom_stopwords)
    
    return ' '.join([w for w in word_list if w not in stop_words])
    
def lemmatization(raw_text):
    word_list = nltk.word_tokenize(raw_text)
    # Init the Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(w) for w in word_list])

def keep_nouns(raw_text):
    """Removes non-nouns words from a string."""
    tokens = nltk.word_tokenize(raw_text)
    tagged = nltk.pos_tag(tokens)

    return ' '.join([token[0] for token in tagged if token[1].startswith('N')])

def tfidf_svd(data, min_df=0.02):
    """Computes tfidf vector representation and reduces dimensionality of the space with SVD.

    Args:
        data (pd.Series): Dataframe column containing the data to be hashed
        min_df (float, optional): Minimum value for df to consider when representing data with tfidf. Defaults to 0.02.

    Returns:
        np.array: Tfidf representation with selected number of components retaining high variance.
    """
    vectorizer = TfidfVectorizer(min_df=min_df)
    tfidf_data = vectorizer.fit_transform(data)
    
    temp_components = tfidf_data.shape[1] - 1 if tfidf_data.shape[1] <= 100 else 100
    svd = TruncatedSVD(n_components=temp_components, random_state=42)
    svd.fit(tfidf_data)
    
    variance = np.cumsum(svd.explained_variance_ratio_)
    n_components = np.argmax(variance >= 0.6) + 1
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    data_reduced = svd.fit_transform(tfidf_data)
    
    return data_reduced