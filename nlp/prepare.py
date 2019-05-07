# text cleaning functions from web scraping
import re
import unicodedata
import json
import pandas as pd
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

def basic_clean(input_str):
    #''' returns a string, lowercase everything, normalize unicode characters, 
    # replace anything that is not a letter, number, whitespace or a single quote '''
    lower_str = input_str.lower()
    normalized = unicodedata.normalize('NFKD', lower_str)
    return_str = re.sub(r'[^\w\s]', ' ', normalized)
    return return_str

def tokenize(input_str):
    #''' returns a string, tokenizes the input string and returns as a string''' 
    tokenizer = nltk.tokenize.ToktokTokenizer()
    tokenizer.tokenize(input_str, return_str=True)
    return_str = tokenizer.tokenize(input_str, return_str=True)
    return return_str

def stem(input_str):
    #''' returns a string, stems the input string and returns as a string'''
    ps = nltk.stem.PorterStemmer()
    return_str = [ps.stem(word) for word in input_str.split()]
    return return_str

def lemmatize(input_str):
     #''' returns a string, stems the input string and returns as a string'''
    wnl = nltk.stem.WordNetLemmatizer()    
    input_list = input_str.split()
    lemmatized_words = [wnl.lemmatize(word) for word in input_list]
    return_str = ' '.join(lemmatized_words)
    return return_str