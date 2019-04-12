import env
import pandas as pd

#   this version uses mysql
def get_mysqlconnection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+mysqlconnector://{user}:{password}@{host}/{db}'

#   this version uses pysql
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_mall_data():
    return pd.read_sql('SELECT * FROM customers;', get_connection('mall_customers'))
