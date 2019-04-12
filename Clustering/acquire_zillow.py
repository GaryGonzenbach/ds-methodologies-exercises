import env
import pandas as pd

#   this version uses mysql
def get_mysqlconnection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+mysqlconnector://{user}:{password}@{host}/{db}'

#   this version uses pysql
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_zillow17(limitrows = -99):
    if limitrows == -99:
        df = pd.read_sql('SELECT * FROM predictions_2017\
            LEFT JOIN properties_2017 USING (parcelid)\
            LEFT JOIN airconditioningtype USING (airconditioningtypeid)\
            LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)\
            LEFT JOIN buildingclasstype USING (buildingclasstypeid)\
            LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)\
            LEFT JOIN propertylandusetype USING (propertylandusetypeid)\
            LEFT JOIN storytype USING (storytypeid)\
            LEFT JOIN typeconstructiontype USING (typeconstructiontypeid);', get_connection('zillow'))
    else:
        df = pd.read_sql('SELECT * FROM predictions_2017\
            LEFT JOIN properties_2017 USING (parcelid)\
            LEFT JOIN airconditioningtype USING (airconditioningtypeid)\
            LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)\
            LEFT JOIN buildingclasstype USING (buildingclasstypeid)\
            LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)\
            LEFT JOIN propertylandusetype USING (propertylandusetypeid)\
            LEFT JOIN storytype USING (storytypeid)\
            LEFT JOIN typeconstructiontype USING (typeconstructiontypeid) LIMIT '+ limitrows +' ;', get_connection('zillow'))
    return df

def get_zillow16(limitrows = -99):
    if limitrows == -99:
        df = pd.read_sql('SELECT * FROM predictions_2016\
            LEFT JOIN properties_2016 USING (parcelid)\
            LEFT JOIN airconditioningtype USING (airconditioningtypeid)\
            LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)\
            LEFT JOIN buildingclasstype USING (buildingclasstypeid)\
            LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)\
            LEFT JOIN propertylandusetype USING (propertylandusetypeid)\
            LEFT JOIN storytype USING (storytypeid)\
            LEFT JOIN typeconstructiontype USING (typeconstructiontypeid);', get_connection('zillow'))
    else:
        df = pd.read_sql('SELECT * FROM predictions_2016\
            LEFT JOIN properties_2016 USING (parcelid)\
            LEFT JOIN airconditioningtype USING (airconditioningtypeid)\
            LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)\
            LEFT JOIN buildingclasstype USING (buildingclasstypeid)\
            LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)\
            LEFT JOIN propertylandusetype USING (propertylandusetypeid)\
            LEFT JOIN storytype USING (storytypeid)\
            LEFT JOIN typeconstructiontype USING (typeconstructiontypeid) LIMIT '+ limitrows +' ;', get_connection('zillow'))
    return df

def clean_zillow_ids(df): 
    df.drop(['airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid',\
     'heatingorsystemtypeid', 'propertylandusetypeid', 'storytypeid','typeconstructiontypeid','id'], inplace=True, axis=1, errors='ignore')
    return df

def drop_rows_missinglatlong(df):
    df.dropna(subset=['latitude'], inplace = True)
    df.dropna(subset=['longitude'], inplace = True)
    return df


def get_zillow_data(limitrows = -99):  
    df2017 = get_zillow17(limitrows) 
    df2016 = get_zillow16(limitrows)  
    df = df2016.append(df2017, ignore_index=True)
    df = clean_zillow_ids(df)
    df = drop_rows_missinglatlong(df)
    return df




