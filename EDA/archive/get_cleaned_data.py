
import pandas as pd
import numpy as np
import pytz
import BachelorProject.ETL.dictionary_mappings as dm
import glob
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
xdrive_path = os.path.join(parent_dir, 'xdrive')
sys.path.append(xdrive_path)
import get_files_from_xdrive as gxdrive



def get_all_weather_files():
     df = gxdrive.read_file_from_xdrive_as_df('WeatherData.csv')
     return df
# def get_data_set_and_split_to_train_and_test():
#     data_set = merge_prices_and_hydro()
#     X_train, X_test, y_train, y_test = train_test_split(data_set.loc[:, data_set.columns != 'Foward Price'], data_set['Forward Price'],
#                                                     random_state=42)
#     return X_train, X_test, y_train, y_test

def get_rain_data_cleaned():
    df_rain = gxdrive.read_file_from_xdrive_as_df('WeatherData.csv')
    df_rain_cleaned = df_rain[['DATA (YYYY-MM-DD)', 'Hora UTC', 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)', 'ESTACAO']]
    stations = gxdrive.read_file_from_xdrive_as_df('stations.csv')

    df_merged = pd.merge(df_rain_cleaned , stations, left_on='ESTACAO', right_on='id_station', how='left')
    df_rain =df_merged.drop(columns=['lat', 'lon', 'city_station', 'state'])
    #CO is Central East which in prices is in SE
    df_rain['region'] = df_rain['region'].replace('CO', 'SE')
    df_rain = df_rain.rename(
        columns = {'DATA (YYYY-MM-DD)':'DateValueUTC', 'Hora UTC': 'TimeValueUTC', 
                'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'Rain', 
                'ESTACAO': 'Metering Station', 'region': 'Region'})
    df_rain['Unit'] = 'mm'
    df_rain['TimeValueUTC'] = df_rain['TimeValueUTC'].str[:2]

    df_temp = df_rain[['DateValueUTC', 'TimeValueUTC', 'Region', 'Rain', 'Unit']]
    df_temp['DateTime'] = pd.to_datetime((df_temp['DateValueUTC']).astype(str) + ' ' + (df_temp['TimeValueUTC']).astype(str), format='%Y-%m-%d %H:%M:%S')
    df_temp['DateTime'] = pd.to_datetime(df_temp['DateTime'], format='%Y-%m-%d %H:%M:%S')
    df_temp['Month'] = df_temp['DateTime'].dt.month
    df_temp['Region'] = df_temp['Region'].replace('CO', 'SE')

    # Define the timezones
    utc = pytz.utc
    cet = pytz.timezone('Europe/Berlin')

    df_temp['DateTime'] = pd.to_datetime((df_temp['DateValueUTC']).astype(str) + ' ' + (df_temp['TimeValueUTC']).astype(str), format='%Y-%m-%d %H:%M:%S')
    df_temp['DateTime'] = pd.to_datetime(df_temp['DateTime'], format='%Y-%m-%d %H:%M:%S')
    # Convert the 'TimeValueUTC' column from UTC to CET
    df_temp['TimeValueCET'] = df_temp['DateTime'].dt.tz_localize('UTC').dt.tz_convert(cet)

    # Create a 'day_of_week' feature
    df_temp['day_of_week'] = df_temp['DateTime'].dt.dayofweek  # Monday=0, Sunday=6

    # Cyclical encoding for hour
    df_temp['hour_sin'] = np.sin(2 * np.pi * df_temp['DateTime'].dt.hour / 24)
    df_temp['hour_cos'] = np.cos(2 * np.pi * df_temp['DateTime'].dt.hour / 24)

    # Cyclical encoding for day of year
    df_temp['day_of_year'] = df_temp['DateTime'].dt.dayofyear
    df_temp['day_sin'] = np.sin(2 * np.pi * df_temp['day_of_year'] / 365)
    df_temp['day_cos'] = np.cos(2 * np.pi * df_temp['day_of_year'] / 365)

    # Optionally: binary feature for weekend
    df_temp['is_weekend'] = df_temp['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    df_temp['year'] = df_temp['DateTime'].dt.year
    # Create a mapping dictionary for subsystem_id to PowerPriceAreaCode
    mapping = {
        'N': 'BRAZIL_NORTH',
        'NE': 'BRAZIL_NORTHEAST',
        'S': 'BRAZIL_SOUTH',
        'SE': 'BRAZIL_SOUTHEAST_CENTRALWEST'
    }

    # Create a new column 'PowerPriceAreaCode' based on the mapping
    df_temp['PowerPriceAreaCode'] = df_temp['Region'].map(mapping)

    return df_temp


def mapp_region_to_power_price_are_code(df1):
    mapping = {
        'N': 'BRAZIL_NORTH',
        'NE': 'BRAZIL_NORTHEAST',
        'S': 'BRAZIL_SOUTH',
        'SE': 'BRAZIL_SOUTHEAST_CENTRALWEST'
        
    }

    # Create a new column 'PowerPriceAreaCode' based on the mapping
    df1['PowerPriceAreaCode'] = df1['Region'].map(mapping)
    return df1


def get_market_price_data_cleaned():
    # Load the data
    #df_market_price = pd.read_excel('./data_sources/Price_Daily.xlsx')
    df_market_price = gxdrive.read_file_from_xdrive_as_df("Price_Daily.xlsx")

    # Consistent column names across data sources
    df_market_price.rename(columns={
        'ReferenceDateTimeOffset': 'DateTime',
        'DeliveryStartDateTimeOffset': 'DeliveryStartDate',
        'PriceMWh': 'Forward Price'
    }, inplace=True)

     # Consistent date format across data sources

    df_market_price['DateTime'] = pd.to_datetime(df_market_price['DateTime']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df_market_price['DeliveryStartDate'] = pd.to_datetime(df_market_price['DeliveryStartDate']).dt.strftime('%Y-%m-%d %H:%M:%S')

    #Extract features from DateTime
    df_market_price['Year'] = pd.to_datetime(df_market_price['DateTime']).dt.year
    df_market_price['Month'] = pd.to_datetime(df_market_price['DateTime']).dt.month
    df_market_price['Day'] = pd.to_datetime(df_market_price['DateTime']).dt.day
    df_market_price['Day Of Week'] = pd.to_datetime(df_market_price['DateTime']).dt.day_name()


    df_market_price.drop(columns=['PriceI5MWh'], inplace=True)

    # Add the 'Region' column and assign the value 'SE/CW' to all rows
    df_market_price['Region'] = 'SE/CW'
    df_market_price['DateTime'] = pd.to_datetime(df_market_price['DateTime'])


    return df_market_price


def get_settlement_price_data_cleaned():

    # Read Price Data Excel file
    
    #df_price = pd.read_excel('./data_sources/Price Data.xlsx')
    df_price = gxdrive.read_file_from_xdrive_as_df("PLDData.xlsx")
    df_price = df_price[['DateValueCET', 'TimeValueCET', 'PowerPriceAreaCode', 'PriceMWh']]
    df_price['DateValueCET'] = pd.to_datetime(df_price['DateValueCET'])
    df_price['TimeValueCET'] = pd.to_timedelta(df_price['TimeValueCET'])
    df_price['DateTime'] = df_price['DateValueCET'] + df_price['TimeValueCET']
    df_price['DateTime'] = pd.to_datetime(df_price['DateTime'], format='%Y-%m-%d %H:%M:%S')
  
    df_price['Day_of_year'] = df_price['DateTime'].dt.dayofyear
    df_price['Hour'] = df_price['DateTime'].dt.hour
    df_price['Year'] = df_price['DateTime'].dt.year

    df_price['Region'] = df_price['PowerPriceAreaCode'].replace(dm.region_dict_power_price_area_code)
    df_price.rename(columns={
    'PriceMWh': 'Settlement Price'
    })

    df_price = df_price.drop(columns=['PowerPriceAreaCode'])
    df_filtered_price = df_price[df_price['Region'] == 'SE/CW']
    df_aggregated_settlement_price = df_filtered_price.groupby('DateTime').agg(
        average_settlement_price =('PriceMWh', 'mean'),
        variance_in_settlement_price=('PriceMWh', 'var'),
        std_settlement_price=('PriceMWh', 'std'),
        min_settlement_price=('PriceMWh', 'min'),
        max_settlement_price=('PriceMWh', 'max'),
        settlement_price_range=('PriceMWh', lambda x: x.max() - x.min()),
        median_settlement_price=('PriceMWh', 'median'),
        first_settlement_price=('PriceMWh', 'first'),
        last_settlement_price=('PriceMWh', 'last'),
        count_price=('PriceMWh', 'count'),
        skewness_settlement_price=('PriceMWh', 'skew')
    ).reset_index()

    df_aggregated_settlement_price['DateTime'] = pd.to_datetime(df_aggregated_settlement_price['DateTime'])


    return df_aggregated_settlement_price


def get_market_and_settlement_price_merged(df_market_price, df_settlemet_price):
    # Merge the two dataframes on the 'DateTime' column
    df_merged = pd.merge(df_market_price, df_settlemet_price, on='DateTime', how='outer')
    return df_merged



def get_wind_data_for_visualization():
    #get_data
    df_weather = get_all_weather_files()
    df_rain_cleaned = df_weather[['DATA (YYYY-MM-DD)', 'Hora UTC',  'VENTO, VELOCIDADE HORARIA (m/s)', 'ESTACAO']]
    #handle station mappins and data cleaning
    df_weather_cropped = df_rain_cleaned[['DATA (YYYY-MM-DD)','Hora UTC', 'VENTO, VELOCIDADE HORARIA (m/s)', 'ESTACAO']].dropna()
    df_weather_c = df_weather_cropped.rename(
                columns = {'DATA (YYYY-MM-DD)':'DateValueUTC', 'Hora UTC': 'TimeValueUTC', 
                        'VENTO, VELOCIDADE HORARIA (m/s)' : 'Average Wind Speed (m/s)',
                        'ESTACAO': 'Metering Station'})
    df_weather_c['TimeValueUTC'] = df_weather_c['TimeValueUTC'].str[:2]
    df_weather_c['DateTime'] = pd.to_datetime((df_weather_c['DateValueUTC']).astype(str) + ' ' + (df_weather_c['TimeValueUTC']).astype(str), format='%Y-%m-%d %H:%M:%S')
    stations = gxdrive.read_file_from_xdrive_as_df("stations.csv")
    stations =stations.drop(columns=['lat', 'lon', 'city_station', 'state', 'record_first', 'record_last', 'lvl'])
    df_merged = pd.merge(df_weather_c , stations, left_on='Metering Station', right_on='id_station', how='left')
    df_merged['region'] = df_merged['region'].replace('CO', 'SE')
    df_merged = df_merged.drop(columns=['Metering Station', 'id_station', 'DateValueUTC', 'TimeValueUTC'])
    df_merged.set_index('DateTime', inplace=True)
    return df_merged

def get_price_and_rain_df(df1, df2):
    all_values =  pd.merge(df1, df2, on=['PowerPriceAreaCode', 'DateTime'], how='outer')
    return all_values


def get_hydro_inflow_cleaned():
    
    df_hydro =  gxdrive.read_file_from_xdrive_as_df("Hydro_Daily.xlsx")
    df_hydro = df_hydro[['DateValueCET', 'PowerPriceAreaCode', 'Value', 'Location']]
    df_hydro['Region'] = df_hydro['Location'].replace(dm.region_dict_bruno)
    df_hydro = df_hydro.rename(columns={'DateValueCET': 'DateTime', 'Value': 'Inflow'})
    df_hydro = df_hydro.drop(['PowerPriceAreaCode', 'Location'], axis=1)
    return df_hydro

def get_market_prices_cleaned():
    df_price = gxdrive.read_file_from_xdrive_as_df("Price_Daily.xlsx")

    # Consistant column names across data sources
    df_price.rename(columns={
        'ReferenceDateTimeOffset': 'DateTime',
        'DeliveryStartDateTimeOffset': 'DeliveryStartDate',
        'PriceMWh': 'Price'
    }, inplace=True)

    # Consistant date format across data sources
    df_price['DateTime'] = pd.to_datetime(df_price['DateTime']).dt.strftime('%Y-%m-%d')
    df_price['DeliveryStartDate'] = pd.to_datetime(df_price['DeliveryStartDate']).dt.strftime('%Y-%m-%d')


    df_price.drop(columns=['PriceI5MWh'], inplace=True)
    # Add the 'Region' column and assign the value 'SE/CW' to all rows
    df_price['Region'] = 'SE/CW'
    return df_price


#this is only for visualization, still on an hourly granularity, in case we want to see it in a heat map and see if there are some patterns
def get_radiation_data_for_visualization():
    #get_data
    df_weather = get_all_weather_files()
    #handle station mappins and data cleaning
    df_weather_cropped = df_weather[['DATA (YYYY-MM-DD)','Hora UTC',  'RADIACAO GLOBAL (KJ/m²)', 'ESTACAO']].dropna()
    df_weather_c = df_weather_cropped.rename(
            columns = {'DATA (YYYY-MM-DD)':'DateValueUTC', 'Hora UTC': 'TimeValueUTC', 
                    'RADIACAO GLOBAL (KJ/m²)': 'Radiation', 
                    'ESTACAO': 'Metering Station'})
    df_weather_c['TimeValueUTC'] = df_weather_c['TimeValueUTC'].str[:2]
    df_weather_c['DateTime'] = pd.to_datetime((df_weather_c['DateValueUTC']).astype(str) + ' ' + (df_weather_c['TimeValueUTC']).astype(str), format='%Y-%m-%d %H:%M:%S')
    stations= gxdrive.read_file_from_xdrive_as_df("stations.csv")
    stations =stations.drop(columns=['lat', 'lon', 'city_station', 'state', 'record_first', 'record_last', 'lvl'])
    df_merged = pd.merge(df_weather_c , stations, left_on='Metering Station', right_on='id_station', how='left')
    df_merged['region'] = df_merged['region'].replace('CO', 'SE')
    df_merged = df_merged.drop(columns=['Metering Station', 'id_station', 'DateValueUTC', 'TimeValueUTC'])
    df_merged.set_index('DateTime', inplace=True)

    return df_merged


# if __name__ == "__main__":
 
#    df = get_market_price_data_cleaned()