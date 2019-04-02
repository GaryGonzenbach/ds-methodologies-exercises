import env
import pandas as pd

#   this version uses mysql
def get_mysqlconnection(db, user=env.user, host=env.host, password=env.pw):
    return f'mysql+mysqlconnector://{user}:{password}@{host}/{db}'

#   this version uses pysql
def get_connection(db, user=env.user, host=env.host, password=env.pw):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_titanic_data():
    return pd.read_sql('SELECT * FROM passengers;', get_connection('titanic_db'))
    
def get_iris_data():
    Sql_str = 'SELECT m.*, s.species_name FROM measurements as m JOIN species as s ON m.species_id=s.species_id'                         
    return pd.read_sql(Sql_str,  get_connection('iris_db')) 
       
 