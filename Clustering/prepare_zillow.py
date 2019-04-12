''' utilities for preparing dataframes'''
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def get_numeric_columns(df, skipcolumns=[]):
    #   ''' arguments - (dataframe, optional list of strings)
    #   Purpose is to return a list of numeric columns from a dataframe
    #   second argument is optional - is a list of numeric columns you dont want included in the returned list '''df.fillna(value=0, inplace=True)   
    column_list = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_column_list = [x for x in column_list if x not in skipcolumns]
    return(numeric_column_list)

def change_numeric_2str(df, columns2change=[]):
    # ''' arguments  - (dataframe, optional list of strings
    # Purpose is to change a specified set of columns anything to a category (categorical column for plotting)
    # send a list of columns to change from numeric to type category (for creating categorical columns in a dataframe)
    # returns the dataframe with the changed columns '''
    columns2change = ['fips','regionidcounty','yearbuilt']
    for column in columns2change:
        df[[column]] = df[[column]].astype('str')
#        df[[column]] = df[[column]].astype('category')
#        df[[column]] = df[[column]].add_categories([0])
    return(df)

def matching_strings_inrows(df,col_name,matchstr):
    # '''  arguments  - (dataframe, string, list of strings) 
    # Purpose is to filter out rows of a dataframe where a specific column contains a text string
    # - can pass it multiple strings by passing in a list of strings
    # you should take care not to pick text strings that are non-unique, 
    # otherwise you could end up with a bigger dataset than you started with'''
    matches = [] 
    total_rows = 0  
    for xstr in matchstr: 
        lower_xstr = xstr.lower()
        strmatch = df[df[col_name].str.lower().str.contains(lower_xstr)]
        total_rows += len(strmatch)
        matches.append(strmatch)
    if total_rows > len(df):
        print("ERROR: non-unique strings resulted in duplicate rows, re-select text strings and try again!!!")    
    return(pd.concat(matches, axis=0))

def delete_missing_byrow(df, zero_delete, empty_delete, null_delete):
    # '''  arguments  - (dataframe, fraction, fraction, fraction)  all fractions between 0 and 1
    # Purpose is to remove rows in a dataframe where the number of zeros, blanks, or nulls
    # - exceeds any of three thresholds passed into the function  
    # - thresholds is the total count of zeros, blanks, or nulls on a given row'''
    total_num_of_columns = len(df.columns)
    df['zero_count'] = df.iloc[:, 1:].eq(0).sum(axis=1)
    df['empty_count'] = df.iloc[:, 1:].eq("").sum(axis=1)
    df['null_count'] = df.isnull().sum(axis=1)
    if null_delete > 0:
        col_threshold = null_delete * total_num_of_columns 
        df.drop(df[df.null_count > col_threshold].index, inplace=True)
    if zero_delete > 0:    
        col_threshold = zero_delete * total_num_of_columns
        df.drop(df[df.zero_count > col_threshold].index, inplace=True)
    if empty_delete > 0:
        col_threshold = empty_delete * total_num_of_columns
        df.drop(df[df.empty_count > col_threshold].index, inplace=True)       
    dropcols = ['zero_count','empty_count','null_count']
    df.drop(dropcols,axis=1,inplace=True)   
    return(df)

def replace_nulls(df):
    df.fillna(value=0, inplace=True)
    return df

def delete_missing_bycolumn(df, threshold):
    # '''  arguments  - (dataframe, fraction from 0-1) 
    # Purpose is to remove columns in a dataframe where the number of  nulls
    # - exceeds the threshold fraction (of total rows) of the dataframe  '''   
    total_rows = len(df)
    null_count = df.isnull().sum()
    column_list = df.columns.tolist()
    for thiscolumn in column_list:
        if null_count[thiscolumn] > (threshold * total_rows):
            df.drop([thiscolumn], axis=1, inplace=True)
    return(df)  

def remove_outliers(df, method, k, listofcolumns):
    print('This routine is todo')
    return(df)

def numeric_filter(df,column,minvalue,maxvalue):
    # ''' return the dataframe where the values in the specified column are between the min and the max 
    df = df[df[column] >= minvalue]
    df = df[df[column] <= maxvalue]
    return(df)

def delete_specific_column(df, column_list):
    for thiscolumn in column_list:
        df.drop([thiscolumn], axis=1, inplace=True)
    return(df)  

def create_clusters(df,col_list,n_clusters,nameof_clustercolumn):
    # '''  pass a datframe, list of columns to cluster, and a target number of clusters
    #      returns the modified datagrame, the inertia of the cluster fit, and a color constant applied to each point
    #   first, remove 'cluster_target' in case calling this function iteratively
    #   function will return a df with the 'cluster_target' column, evaluated and appended to it 
    #  '''
    if nameof_clustercolumn in df.columns:     
        df.drop([nameof_clustercolumn],axis=1,inplace=True)
    cluster_df = df.copy()
    df_columns = df.columns
    for thiscolumn in df_columns:
        if thiscolumn not in col_list:
            cluster_df.drop([thiscolumn], axis=1, inplace=True)
    kmeans = KMeans(n_clusters)            
    kmeans.fit(cluster_df)
    return_inertia = kmeans.inertia_ 
    return_labels = kmeans.labels_       
    cluster_df[nameof_clustercolumn] = kmeans.predict(cluster_df)        
    cluster_df.drop(col_list,axis=1,inplace=True)
    return_df = pd.concat([df,cluster_df], axis=1, join_axes=[df.index]) 
    return return_df, return_inertia, return_labels

#
#    the following functions are zillow specific 
#  
def manually_remove_outliers_from_zillow(df):
    keys = ['bathroomcnt','bedroomcnt','calculatedfinishedsquarefeet','structuretaxvaluedollarcnt','landtaxvaluedollarcnt']
    values = [(1,7), (1,7), (500,8000), (25000,2000000), (10000,2500000)]
    dictionary = dict(zip(keys, values))
    for key, value in dictionary.items():         
        df = numeric_filter(df,key,value[0],value[1])
#        df = df[df[key] >= value[0]]
#        df = df[df[key] <= value[1]]
    return(df)

def prepare_zillow_db(df):
    # '''  arguments  - (dataframe)
    # order is important , for example - 'propertylandusedesc' is an important filter to apply on the database
    # which could possibly be automatically eliminated with 'delete_missing_by_column', depending on what thresholds are used  
    # so call it first'''
    text_list = ['single', 'condominium']
    df = matching_strings_inrows(df, 'propertylandusedesc', text_list )
    df = manually_remove_outliers_from_zillow(df)
    df = delete_specific_column(df, ['roomcnt','taxamount','fullbathcnt'])
    df = delete_missing_bycolumn(df, 0.2)   
    df = delete_missing_byrow(df, .1, .1, .1)
    col_list = ['fips','regionidcounty','yearbuilt']
    df = change_numeric_2str(df, col_list)
    df = replace_nulls(df)
    return(df)
