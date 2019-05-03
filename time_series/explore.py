''' utilities for preparing dataframes'''
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser
from sklearn.cluster import KMeans
import scipy.stats as stats
from scipy.stats import pearsonr
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

# '''     column manipulations       '''

def replace_nulls(df):
    df.fillna(value=0, inplace=True)
    return df

def combine_two_columns(df, newcolumn_name, column1, column2, deleteold = False):
    #   ''' arguments - (dataframe, newcolumn name (str), column1 (str), column2 (str), optional bool to delete the two old columns
    #  '''       
    df[newcolumn_name] = df[column1] + df[column2]
    if deleteold == True:
        df.drop([column1], axis=1, inplace=True)
        df.drop([column2], axis=1, inplace=True)
    return df

def get_numeric_columns(df, skipcolumns=[]):
    #   ''' arguments - (dataframe, optional list of strings)
    #   Purpose is to return a list of numeric columns from a dataframe
    #   second argument is optional - is a list of numeric columns you dont want included in the returned list '''df.fillna(value=0, inplace=True)   
    column_list = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_column_list = [x for x in column_list if x not in skipcolumns]
    return(numeric_column_list)

def filter_columns(df,listofcolumns):
    #   ''' arguments - (dataframe), columns to include in returned dataframe
    #  ''' 
    newdf = df.copy()
    col_list = df.columns 
    for column in col_list:
        if column not in listofcolumns:
            newdf.drop([column], axis=1, inplace=True)
    return newdf


def change_numeric_2str(df, columns2change=[]):
    # ''' arguments  - (dataframe, optional list of strings
    # Purpose is to change a specified set of columns anything to a category (categorical column for plotting)
    # send a list of columns to change from numeric to type category (for creating categorical columns in a dataframe)
    # returns the dataframe with the changed columns '''
    for column in columns2change:
        df[[column]] = df[[column]].astype('str')
#        df[[column]] = df[[column]].astype('category')
#        df[[column]] = df[[column]].add_categories([0])
    return(df)

def delete_missing_bycolumn(df, threshold, skipcolumns=[] ):
    # '''  arguments  - (dataframe, fraction from 0-1, columns to skip (list) is optional) 
    # Purpose is to remove columns in a dataframe where the number of  nulls
    # - exceeds the threshold fraction (of total rows) of the dataframe  '''   
    total_rows = len(df)
    null_count = df.isnull().sum()
    column_list = df.columns.tolist()
    for thiscolumn in column_list:
        if null_count[thiscolumn] > (threshold * total_rows):
            if thiscolumn not in skipcolumns:
                df.drop([thiscolumn], axis=1, inplace=True)
    return(df)  

def delete_specific_column(df, column_list):
    for thiscolumn in column_list:
        df.drop([thiscolumn], axis=1, inplace=True)
    return(df)  



# '''      row manipulations    '''

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

def remove_outliers(df, method, k, listofcolumns):
    print('This routine is todo')
    return(df)

def numeric_filter(df,column,minvalue,maxvalue):
    # ''' return the rows in the dataframe where the values in the specified column are between the min and the max 
    df = df[df[column] >= minvalue]
    df = df[df[column] <= maxvalue]
    return(df)



def compare_chitests(df,skipcolumns=[]):
    # '''    ''' 
    numeric_cols = get_numeric_columns(df)
    str_columns = []
    allcolumns = df.columns
    for thiscolumn in allcolumns:
        if thiscolumn not in numeric_columns:
            if thiscolumn not in skipcolumns:
                str_columns.append(thiscolumn)
    return


# '''     Stats utils  '''

def compare_ttests(df,dependentvar,skipcolumns=[]):
    # '''  arguments  - (dataframe, dependent variable column, columns to skip (list) is optional) 
    # Purpose is to run Ttests on the dependent variable value of all columns, separated by the mean of each column
    # Only results where pvalue is less than 0.05 are returned (others are dropped)
    # returns a dataframe sorted by the coefficient of the ttest for each variable  '''   
    tcolumns = get_numeric_columns(df, skipcolumns=[])
    results = []
    features = []
    pvalues = []
    for thiscolumn in tcolumns:
        if thiscolumn != dependentvar:
            if thiscolumn != skipcolumns:
                df1 = df[df[thiscolumn] < df[thiscolumn].mean()]
                df2 = df[df[thiscolumn] > df[thiscolumn].mean()]        
                this_tstat, this_pvalue = stats.ttest_ind(df1[dependentvar].dropna(),
                   df2[dependentvar].dropna())
                if this_pvalue < .05:
                    results.append(this_tstat)
                    features.append(thiscolumn)
                    pvalues.append(this_pvalue)    
    list_of_tuples = list(zip(features,results,pvalues))
    results_df = pd.DataFrame(list_of_tuples, columns = ['Variables','T-Stats','Pvalues'])
    results_df.sort_values('T-Stats', inplace=True)
    return results_df    

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

def regressiontest(df,xfeatures,yfeature,splitfraction):
    y = df[yfeature]
    X = filter_columns(df,xfeatures)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=splitfraction, random_state=123)
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
#
    column_names = X_train.columns
    r_and_p_values = [pearsonr(X_train[col], y_train) for col in column_names]
    corrdict = dict(zip(column_names, r_and_p_values))
#
    ols_model = sm.OLS(y_train, X_train)
    fit = ols_model.fit()
    lm1 = LinearRegression(fit_intercept=False) 
    lm1.fit(X_train[xfeatures], y_train)
    LinearRegression(copy_X=True, fit_intercept=False, n_jobs=None,
         normalize=False)
    lm1_y_intercept = lm1.intercept_
    lm1_coefficients = lm1.coef_
    coeff_list = list(zip(xfeatures,lm1.coef_))
    y_pred_lm1 = lm1.predict(X_train[xfeatures])
    mse = mean_squared_error(y_train, y_pred_lm1)
    r2 = r2_score(y_train, y_pred_lm1)
    return mse, r2, corrdict, coeff_list
#
