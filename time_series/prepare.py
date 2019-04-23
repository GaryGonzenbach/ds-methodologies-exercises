# prepare utils
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser


def convert_to_datetime(df,thiscolumn,useUTC=False,createsubdates=False):
    current_columns = df.columns.tolist()
    thiscolumn = 'datetime_temp'
    if this column in current_columns:
        df.drop([thiscolumn], axis=1, inplace=True) 
    df['datetime_temp'] =  pd.to_datetime(df[thiscolumn], utc=useUTC)

resume right here,    pull code fro, parse_accesslog   line 8

#  df = df.tz_localize('utc').tz_convert('America/Chicago')
    current_columns = 
    df.drop([thiscolumn], axis=1, inplace=True)
    df.rename({'datetime_temp':thiscolumn}, axis=1, inplace=True)
    if createsubdates == True:

        df.datetime.dt.hour

    return df