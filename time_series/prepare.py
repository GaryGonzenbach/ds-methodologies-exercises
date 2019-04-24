# prepare utils
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser


def convert_to_datetime(df,column2convert,useUTC=False,createsubdates=[]):
    # ''' pass it a dataframe and the name of the column to convert to datetime format
    #   optionally pass it a paramenter to use UTC (which will also localize it to CMT
    #   after setting the dataframe index to it )
    #   and also optionally pass it a list of datetime sub-columns "M", "D", "Y", "Qtr"
    #  '''
    current_columns = df.columns.tolist()
    #   set up temporary column, which will be removed at the end of the function
    thiscolumn = 'datetime_temp'
    if thiscolumn in current_columns:
        df.drop([thiscolumn], axis=1, inplace=True) 
    df['datetime_temp'] =  pd.to_datetime(df[column2convert])
    df.drop([column2convert], axis=1, inplace=True)
    df.rename(columns={'datetime_temp':column2convert},inplace=True)
    df.set_index(column2convert,inplace=True)
    if useUTC == True:
        df = df.tz_localize('utc').tz_convert('America/Chicago')
    for datecomponent in createsubdates:
        if (datecomponent.lower == 'd' or datecomponent.lower == 'day'):
            df['day'] = df.date.dt.day
        elif (datecomponent.lower == 'm' or datecomponent.lower == 'month'):    
            df['month'] = df.date.dt.month
        elif (datecomponent.lower == 'y' or datecomponent.lower == 'year'):    
            df['year'] = df.date.dt.year
        elif datecomponent.lower == 'dayofweek':    
            df['dayofweek'] = df.date.dt.dayofweek
        elif datecomponent.lower == 'dayofyear':    
            df['dayofyear'] = df.date.dt.dayofyear
        elif (datecomponent.lower == 'h' or datecomponent.lower == 'hour'):    
            df['month'] = df.date.dt.month
        elif (datecomponent.lower == 's' or datecomponent.lower == 'second'):    
            df['second'] = df.date.dt.second
        elif (datecomponent.lower == 'qtr' or datecomponent.lower == 'quarter'):    
            df['quarter'] = df.date.dt.quarter
    return df