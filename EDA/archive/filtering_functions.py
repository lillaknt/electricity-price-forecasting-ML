import pandas as pd 
import datetime as dt

def get_values_for_specific_year(df, year:int):
    df_x_year = df[df['DateTime'].dt.year == year]
    return df_x_year