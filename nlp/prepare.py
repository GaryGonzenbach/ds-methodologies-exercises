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
    #''' input a string, returns a list, stems the input string and returns as a list'''
    ps = nltk.stem.PorterStemmer()
    stem_list = [ps.stem(word) for word in input_str.split()]
    return_str = ' '.join(stem_list)
    return return_str

def lemmatize(input_str):
     #''' returns a string, stems the input string and returns as a string'''
    wnl = nltk.stem.WordNetLemmatizer()    
    input_list = input_str.split()
    lemmatized_words = [wnl.lemmatize(word) for word in input_list]
    return_str = ' '.join(lemmatized_words)
    return return_str

def remove_stopwords(input_str, add_stopwords=[], exclude_stopwords=[]):
    #''' returns a string, optionally add a list of words to the stopword list
    #  , also optionally add a list of words to exclude from stopwords 
    stopwords = nltk.corpus.stopwords.words('english')
    if len(add_stopwords) > 0:
        stopwords.extend(add_stopwords)        
    if len(exclude_stopwords) > 0:
        final_stop_words = [word for word in stopwords if word not in exclude_stopwords]   
    else:
        final_stop_words = stopwords
    stopwords = final_stop_words
    input_list = input_str.split()
    without_stopwords = [word for word in input_list if word not in stopwords]        
    return_str = ' '.join(without_stopwords)        
    return return_str

def pull_article_asdict(df,rownum):
    #''' pass a dataframe and row, pull an article and title from the articles dataframe
    #    return it as a dictionary'''
    article = df.at[rownum,'contents']
    title = df.at[rownum,'title']
    article_dict = {'title' : title, 'contents' : article}
    return article_dict

def prep_article(article_dict):
    #''' pass it a dictionary, pulls out the contents text, and returns a dictionary  with text 
    # . that has been stemmed, lemmatized, and cleaned ''''''
    title = article_dict['title']
    original_text = article_dict['contents']
    stemmed_text = stem(original_text)
    lemmatized_text = lemmatize(original_text)
    cleaned_text = remove_stopwords(basic_clean(lemmatized_text))
    return_dict = {'title' : title, 'original' : original_text, 'stemmed' : stemmed_text, 'lemmatized' : lemmatized_text, 'clean' : cleaned_text}
    return return_dict

def prepare_article_data(df):
    #''' pass a dataframe
    #    returns a list of dictionary'''
    cleaned_dicts = []
    for x in range(len(df)):
        rownum = x - 1
        contents = df.at[x,'contents']
        title = df.at[x,'title']
        article_dict = {'title' : title, 'contents' : contents}
        cleaned_dict = prep_article(article_dict)
        cleaned_dicts.append(cleaned_dict)
    return cleaned_dicts

def clean(text):
   return remove_stopwords(lemmatize(basic_clean(text)))

def pull_article_astext(df,rownum):
    #''' pass a dataframe and row, pull an article and title from the articles dataframe
    #    return it as a dictionary'''
    article = df.at[rownum,'cleaned_text']
    return article