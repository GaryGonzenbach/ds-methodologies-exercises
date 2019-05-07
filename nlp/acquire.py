import pandas as pd
from requests import get
from bs4 import BeautifulSoup
import os

def make_soup(url,class_str):
    headers = {'User-Agent': 'Codeup Ada Data Science'} # codeup.com doesn't like our default user-agent
    response = get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.title
    article = soup.find('div', class_= class_str)
    article_text = article.text 
    article_dict = {'title':  title, 'contents': article_text}
    return article_dict



def get_Codeup_blog_articles(class_str = 'mk-single-content'):
    list_of_dicts = []
    list_of_urls = ['https://codeup.com/codeups-data-science-career-accelerator-is-here/',\
        'https://codeup.com/data-science-myths/',\
        'https://codeup.com/data-science-vs-data-analytics-whats-the-difference/', \
        'https://codeup.com/10-tips-to-crush-it-at-the-sa-tech-job-fair/', \
        'https://codeup.com/competitor-bootcamps-are-closing-is-the-model-in-danger/'] 
    for url in list_of_urls:
        new_dict = make_soup(url,class_str)
        list_of_dicts.append(dict(new_dict))
    web_articles = pd.DataFrame(list_of_dicts)
#  rearrange the order of the columns
    web_articles = web_articles[['title', 'contents']]
# retuns a dataframe    
    return web_articles

if __name__ == "__main__":
    # execute only if run as a script
    blog_articles = get_Codeup_blog_articles()
    blog_articles.head()
    
