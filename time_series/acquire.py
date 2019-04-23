import os
import pandas as pd
import numpy as np
from datetime import datetime
import itertools

# JSON API
import requests
import json

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def  getweb_pagerange(base_url,category):
    # ''' pass it the base url str and the category string
    #   reads the page count and returns the max number of pages in that categoiry '''
    url_str = base_url + '/api/v1/' + category  
    response = requests.get(url_str)
    data = response.json()
    print('category: ',category,' max_page: %s' % data['payload']['max_page'])
#   print('next_page: %s' % data['payload']['next_page'])
    return data['payload']['max_page']
    

def  getwebdata(base_url,numberofpages,category):
    # ''' pass it the base url, number of pages to read, and the category,
    #    returns a dataframe  '''
    pagedata = []
    url_addr = base_url + '/api/v1/' + category +'?page='
    for i in range(1,numberofpages+1):
        response = requests.get(url_addr+str(i))
        data = response.json()
        newpage = pd.DataFrame(data['payload'][category])
        pagedata.append(newpage)
        print('Loading ',category,'  page:',str(i))
    df = pd.concat(pagedata, axis=0)    
    return(df)   
        
def acquire_zachdata():
    base_url = 'https://python.zach.lol' 
    if os.path.exists('Items_dZach.csv'):
        print('Reading Items from local csv')
        df_items = pd.read_csv('Items_dZach.csv')
    else:
        max_itempage = getweb_pagerange(base_url,'items')
#        url_str = base_url + '/api/v1/' + category +'?page='
        df_items = getwebdata(base_url,max_itempage,'items')
        df_items.to_csv('Items_dZach.csv',index=False)
    if os.path.exists('Stores_dZach.csv'):
        print('Reading Stores from local csv')
        df_stores = pd.read_csv('Stores_dZach.csv')
    else:
        max_storepage = getweb_pagerange(base_url,'stores')
#        df_stores = getwebdata('https://python.zach.lol/api/v1/stores?page=',max_storepage,'stores')
        df_stores = getwebdata(base_url,max_storepage,'stores')
        df_stores.to_csv('Stores_dZach.csv',index=False)
    if os.path.exists('Sales_dZach.csv'):
        print('Reading Sales from local csv')
        df_sales = pd.read_csv('Sales_dZach.csv')        
    else:
        max_salespage = getweb_pagerange(base_url,'sales')
#        df_sales = getwebdata('https://python.zach.lol/api/v1/sales?page=',max_salespage,'sales')
        df_sales = getwebdata(base_url,max_salespage,'sales') 
        df_sales.to_csv('Sales_dZach.csv',index=False)
    df_sales.rename({'item':'item_id'}, axis=1, inplace=True)
    df_sales.rename({'store':'store_id'}, axis=1, inplace=True) 
    print('Merging all contents to a single dataframe')
    df_sales.merge(df_items, how='inner', on='item_id') 
    df_sales.merge(df_stores, how='inner', on='store_id')
    print('Writing all contents to csv: Total_dZach.csv')
    df_sales.to_csv('Total_dZach.csv',index=False)
    return df_sales