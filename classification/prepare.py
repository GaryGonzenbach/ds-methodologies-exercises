# preparation functions
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

def encode_embarked(df):
#  look for any nulls or 'Na' in this column
    df.embarked.fillna('Other', inplace=True)
    encoder = LabelEncoder()
    encoder.fit(df.embarked)
    return df.assign(embarked_encode = encoder.transform(df.embarked))

#  drop any rows that have null values
def handle_missing_values(df):
    df.dropna(how='any',axis=0, inplace=True)    
    return df

def remove_columns(df):
    return df.drop(columns=['deck'])

def scale_minmax(df):
    scaler = MinMaxScaler()
    scaler.fit(df[['fare']])
    df.fare = scaler.transform(df[['fare']])
    scaler = MinMaxScaler()
    scaler.fit(df[['age']])
    df.age = scaler.transform(df[['age']])
    return df

def encode_sex(df):
#  look for any nulls or 'Na' in this column
    df.sex.fillna('Other', inplace=True)
    encoder = LabelEncoder()
    encoder.fit(df.sex)
    return df.assign(sex_encode = encoder.transform(df.sex))    

def prep_titanic_data(df):
    df = df\
       .pipe(remove_columns)\
       .pipe(encode_embarked)\
       .pipe(encode_sex)\
       .pipe(handle_missing_values)\
       .pipe(scale_minmax)            
    return df

def remove_iriscolumns(df):
    return df.drop(columns=['species_id'])

def rename_iriscolumns(df):
    df.columns = [col.lower().replace('.', '_') for col in df]
    df.rename(columns={'species_name':'species'}, inplace=True)
    return df
#     return 

def encode_species(df):
#  look for any nulls or 'Na' in this column
    df.species_name.fillna('Other', inplace=True)
    encoder = LabelEncoder()
    encoder.fit(df.embarked)
    return df.assign(species_encode = encoder.transform(df.species_name))

def prep_iris_data(df):
    df = df\
       .pipe(rename_iriscolumns)\
       .pipe(remove_iriscolumns)               
    return df
