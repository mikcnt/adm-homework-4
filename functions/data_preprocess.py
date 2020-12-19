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

def clean_data(raw_text):

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