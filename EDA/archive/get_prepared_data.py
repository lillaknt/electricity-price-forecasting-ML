import get_cleaned_data as gcd
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pytz
import BachelorProject.ETL.dictionary_mappings as dm
import pandas as pd
import glob
import os
import sys
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Move up one directory level from EDA to the parent directory
parent_dir = os.path.dirname(current_dir)
# Construct the path to the xdrive folder
xdrive_path = os.path.join(parent_dir, 'xdrive')
# Add the xdrive path to sys.path
sys.path.append(xdrive_path)

# Now you can import your module from xdrive
import get_files_from_xdrive as gxdrive

def get_all_weather_files_prepared():

    df = gxdrive.read_file_from_xdrive_as_df('WeatherData.csv')
    return df

def get_data_set_and_split_to_train_and_test():
    data_set = merge_prices_and_hydro()
    X_train, X_test, y_train, y_test = train_test_split(data_set.loc[:, data_set.columns != 'Foward Price'], data_set['Forward Price'],
                                                    random_state=42)
    return X_train, X_test, y_train, y_test


def get_hydro_inflow_cleaned():
    df_hydro = gxdrive.read_file_from_xdrive_as_df("Hydro_Daily.xlsx")
    df_hydro = df_hydro[['DateValueCET', 'PowerPriceAreaCode', 'Value', 'Location']]
    df_hydro['Region'] = df_hydro['Location'].replace(dm.region_dict_bruno)
    df_hydro = df_hydro.rename(columns={'DateValueCET': 'DateTime', 'Value': 'Inflow'})
    df_hydro = df_hydro.drop(['PowerPriceAreaCode', 'Location'], axis=1)
    df_agg = df_hydro.groupby(['DateTime', 'Region'], as_index=False)['Inflow'].sum()
    # Pivot the DataFrame to get separate columns for each region
    df_pivot = df_agg.pivot(index='DateTime', columns='Region', values='Inflow').reset_index()
    # Add a new column for aggregated values (sum of all regions)
    df_pivot['Total Daily Inflow'] = df_pivot[['N', 'NE', 'S', 'SE/CW']].sum(axis=1)
    df_pivot.columns.names = [None]
    df_pivot.rename(columns=lambda x: 'Inflow_' + x if x in ['N', 'NE', 'S', 'SE/CW'] else x, inplace=True)

    return df_pivot

def get_market_price_data_cleaned():
    # Load the data
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
    df_market_price['DeliveryStartDate']=pd.to_datetime(df_market_price['DeliveryStartDate'])


    return df_market_price


def get_settlement_price_data_cleaned():

    # Read Price Data Excel file
    df_price = gxdrive.read_file_from_xdrive_as_df("PLDData.xlsx")
    df_price = df_price[['DateValueCET', 'TimeValueCET', 'PowerPriceAreaCode', 'PriceMWh']]
    df_price['DateValueCET'] = pd.to_datetime(df_price['DateValueCET'])
    df_price['TimeValueCET'] = pd.to_timedelta(df_price['TimeValueCET'])
    df_price['DateTime'] = df_price['DateValueCET'] + df_price['TimeValueCET']
    df_price['DateTime'] = pd.to_datetime(df_price['DateTime'], format='%Y-%m-%d %H:%M:%S')
  
    df_price['Day_of_year'] = df_price['DateTime'].dt.dayofyear
    df_price['Hour'] = df_price['DateTime'].dt.hour
    df_price['Year'] = df_price['DateTime'].dt.year
    df_price['Date'] = df_price['DateTime'].dt.date

    df_price['Region'] = df_price['PowerPriceAreaCode'].replace(dm.region_dict_power_price_area_code)
    df_price.rename(columns={
    'PriceMWh': 'Settlement Price'
    })

    #

    df_price = df_price.drop(columns=['PowerPriceAreaCode'])
    df_filtered_price = df_price[df_price['Region'] == 'SE/CW']
    df_aggregated_settlement_price = df_filtered_price.groupby('Date').agg(
        average_settlement_price=('PriceMWh', 'mean'),
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

    df_aggregated_settlement_price['DateTime'] = pd.to_datetime(df_aggregated_settlement_price['Date'])


    return df_aggregated_settlement_price


def merge_market_and_settlement_price_merged():
    df_market_price = get_market_price_data_cleaned()
    df_settlemet_price = get_settlement_price_data_cleaned()
    df_merged = pd.merge(df_market_price, df_settlemet_price, on='DateTime', how='inner')
    return df_merged


def merge_prices_and_hydro():

    df_prices = merge_market_and_settlement_price_merged()
    df_hydro = get_hydro_inflow_cleaned()
    df_all = pd.merge(df_prices, df_hydro, on='DateTime', how='inner')
    return df_all


def get_consumption_data_cleaned():
    df_consumption = gxdrive.read_file_from_xdrive_as_df("Consumption_Data.xlsx")
    df_consumption = df_consumption[['DateForecastCET','DateValueCET','VolumeMWh','PowerPriceAreaCode']]
    df_consumption.rename(columns={
        'DateForecastCET': 'DateTime',
        'DateValueCET': 'ForecastedDate',
        'VolumeMWh': 'Consumption'
    }, inplace=True)

    df_consumption['Region'] = df_consumption['PowerPriceAreaCode'].replace(dm.region_dict_power_price_area_code)
    df_consumption = df_consumption.drop(['PowerPriceAreaCode'], axis=1)
    df_agg = df_consumption.groupby(['DateTime', 'ForecastedDate', 'Region'], as_index=False)['Consumption'].sum()
    # Pivot the DataFrame to get separate columns for each region
    df_pivot = df_agg.pivot(index=['DateTime', 'ForecastedDate'], columns='Region', values='Consumption').reset_index()
    # Add a new column for aggregated values (sum of all regions)
    df_pivot['Total Consumption(MWh)'] = df_pivot[['N', 'NE', 'S', 'SE/CW']].sum(axis=1)
    df_pivot.columns.names = [None]
    df_pivot.rename(columns=lambda x: 'Consumption(MWh)_' + x if x in ['N', 'NE', 'S', 'SE/CW'] else x, inplace=True)

    return df_pivot


def get_daily_consumption_data_cleaned():
    df_consumption = gxdrive.read_file_from_xdrive_as_df("Consumption_Data.xlsx")
    df_consumption = df_consumption[['DateForecastCET','DateValueCET','VolumeMWh','PowerPriceAreaCode', 'ForecastRangeCET' ]]
    df_consumption.rename(columns={
        'DateForecastCET': 'DateTime',
        'DateValueCET': 'ForecastedDate',
        'VolumeMWh': 'Consumption',
    }, inplace=True)

    df_consumption['Region'] = df_consumption['PowerPriceAreaCode'].replace(dm.region_dict_power_price_area_code)
    consumption_df = df_consumption[(df_consumption['DateTime'] >= '2019-01-01') & 
                                (df_consumption['ForecastedDate'] >= '2019-01-01')]
    # Ensure the data is sorted by ForecastedDate and DateTime
    consumption_df = consumption_df.sort_values(by=['ForecastedDate', 'DateTime'])

# Filter to keep only the most recent forecast (DateTime <= ForecastedDate)
    most_recent_forecasts = consumption_df[consumption_df['DateTime'] <= consumption_df['ForecastedDate']]

# Select the last forecast for each ForecastedDate
    most_recent_forecasts = most_recent_forecasts.groupby(['ForecastedDate', 'Region'], as_index=False).last()
    # Shift ForecastedDate to compute the number of days to the next forecast
    most_recent_forecasts['NextForecastedDate'] = most_recent_forecasts.groupby('Region')['ForecastedDate'].shift(-1)

# Compute the number of days between the current ForecastedDate and the next ForecastedDate
    most_recent_forecasts['DaysUntilNextForecast'] = (most_recent_forecasts['NextForecastedDate'] - 
                                                  most_recent_forecasts['ForecastedDate']).dt.days

# Fill in any missing values in DaysUntilNextForecast with a reasonable assumption
    most_recent_forecasts['DaysUntilNextForecast'].fillna(7, inplace=True)  # Assuming 1 day for the last forecast

# Compute the daily consumption by dividing the total consumption by the number of days in the forecast period
    most_recent_forecasts['DailyConsumption'] = (most_recent_forecasts['Consumption'] / 
                                             most_recent_forecasts['DaysUntilNextForecast'])
    # Fill missing NextForecastedDate values
    most_recent_forecasts['NextForecastedDate'] = most_recent_forecasts['NextForecastedDate'].fillna(
    most_recent_forecasts['ForecastedDate'] + pd.Timedelta(days=1)  # Set to 1 day after ForecastedDate
    )

    def expand_to_daily(row):
        return pd.DataFrame({
        'DateTime': pd.date_range(start=row['ForecastedDate'], end=row['NextForecastedDate'] - pd.Timedelta(days=1), freq='D'),
        'DailyConsumption': row['DailyConsumption'],
        'Region': row['Region'],
        'OriginalDateTime': row['DateTime']
    })
    # Apply this function to each row to get daily data
    daily_consumption_df = pd.concat(most_recent_forecasts.apply(expand_to_daily, axis=1).to_list(), ignore_index=True)

    df_agg = daily_consumption_df.groupby(['DateTime', 'Region'], as_index=False)['DailyConsumption'].sum()
    
    # Pivot the DataFrame to get separate columns for each region
    df_pivot = df_agg.pivot(index=['DateTime'], columns='Region', values='DailyConsumption').reset_index()
    # Add a new column for aggregated values (sum of all regions)
    df_pivot['Total Consumption(MWh)'] = df_pivot[['N', 'NE', 'S', 'SE/CW']].sum(axis=1)
    df_pivot.columns.names = [None]
    df_pivot.rename(columns=lambda x: 'Consumption(MWh)_' + x if x in ['N', 'NE', 'S', 'SE/CW'] else x, inplace=True)

    return df_pivot


def merge_with_consumption():

    df_prices_hydro = merge_prices_and_hydro()
    df_consumption = get_daily_consumption_data_cleaned()
    df_all = pd.merge(df_prices_hydro, df_consumption, on='DateTime', how='inner')
    return df_all


def merge_prices_hydro_wind_radiation():

    df_prices = merge_market_and_settlement_price_merged()
    df_hydro = get_hydro_inflow_cleaned()
    df_wind = get_wind_data_cleaned()
    df_radiation = get_radiation_data_cleaned()
    df_consumption = get_consumption_data_cleaned()
   # Ensure all DataFrames have DateTime set as the index
    df_prices.index = pd.to_datetime(df_prices['DateTime'])
    df_hydro.set_index('DateTime', inplace=True)
    df_consumption.set_index('DateTime', inplace = True)
    merged_df = df_prices.merge(df_hydro, left_index=True, right_index=True, how='inner') \
                            .merge(df_wind, left_index=True, right_index=True, how='inner') \
                            .merge(df_radiation, left_index=True, right_index=True, how='inner')\
                            .merge(df_consumption, left_index=True, right_index=True, how='inner')
    
    return merged_df

def get_radiation_data_cleaned():
    #get_data
    df_weather = get_all_weather_files_prepared()
    #handle station mappins and data cleaning
    df_weather_cropped = df_weather[['DATA (YYYY-MM-DD)','Hora UTC',  'RADIACAO GLOBAL (KJ/m²)', 'ESTACAO']].dropna()
    df_weather_c = df_weather_cropped.rename(
            columns = {'DATA (YYYY-MM-DD)':'DateValueUTC', 'Hora UTC': 'TimeValueUTC', 
                    'RADIACAO GLOBAL (KJ/m²)': 'Radiation', 
                    'ESTACAO': 'Metering Station'})
    df_weather_c['TimeValueUTC'] = df_weather_c['TimeValueUTC'].str[:2]
    df_weather_c['DateTime'] = pd.to_datetime((df_weather_c['DateValueUTC']).astype(str) + ' ' + (df_weather_c['TimeValueUTC']).astype(str), format='%Y-%m-%d %H:%M:%S')
    stations = gxdrive.read_file_from_xdrive_as_df("stations.csv")
    stations =stations.drop(columns=['lat', 'lon', 'city_station', 'state', 'record_first', 'record_last', 'lvl'])
    df_merged = pd.merge(df_weather_c , stations, left_on='Metering Station', right_on='id_station', how='left')
    df_merged['region'] = df_merged['region'].replace('CO', 'SE')
    df_merged = df_merged.drop(columns=['Metering Station', 'id_station', 'DateValueUTC', 'TimeValueUTC'])
    df_merged.set_index('DateTime', inplace=True)

    #pivot data, to be daily granularity and in 5 separate columns
    df_daily = df_merged.groupby('region').resample('D').sum().reset_index()
    pivot_df = df_daily.pivot(index='DateTime', columns='region', values='Radiation')
    pivot_df['Total_Daily_Radiation (KJ/m2)'] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.rename(
            columns = {'N': 'Radation_N (KJ/m2)', 'NE': 'Radiation_NE (KJ/m2)', 
                    'S': 'Radiation_S (KJ/m2)', 'SE': 'Radiation_SE (KJ/m2)'})
    #pivot_df['DateTime'] = pivot_df.index
    return pivot_df

def get_wind_data_cleaned():
    #get_data
    df_rain = get_all_weather_files_prepared()
    df_rain_cleaned = df_rain[['DATA (YYYY-MM-DD)', 'Hora UTC',  'VENTO, VELOCIDADE HORARIA (m/s)', 'ESTACAO']]
    #handle station mappins and data cleaning
    df_weather_cropped = df_rain_cleaned[['DATA (YYYY-MM-DD)','Hora UTC', 'VENTO, VELOCIDADE HORARIA (m/s)', 'ESTACAO']].dropna()
    df_weather_c = df_weather_cropped.rename(
                columns = {'DATA (YYYY-MM-DD)':'DateValueUTC', 'Hora UTC': 'TimeValueUTC', 
                        'VENTO, VELOCIDADE HORARIA (m/s)' : 'Average Wind Speed (m/s)',
                        'ESTACAO': 'Metering Station'})
    df_weather_c['TimeValueUTC'] = df_weather_c['TimeValueUTC'].str[:2]
    df_weather_c['DateTime'] = pd.to_datetime((df_weather_c['DateValueUTC']).astype(str) + ' ' + (df_weather_c['TimeValueUTC']).astype(str), format='%Y-%m-%d %H:%M:%S')
    stations= gxdrive.read_file_from_xdrive_as_df("stations.csv")
    stations =stations.drop(columns=['lat', 'lon', 'city_station', 'state', 'record_first', 'record_last', 'lvl'])
    df_merged = pd.merge(df_weather_c , stations, left_on='Metering Station', right_on='id_station', how='left')
    df_merged['region'] = df_merged['region'].replace('CO', 'SE')
    df_merged = df_merged.drop(columns=['Metering Station', 'id_station', 'DateValueUTC', 'TimeValueUTC'])
    df_merged.set_index('DateTime', inplace=True)
        #pivot data, to be daily granularity and in 5 separate columns
    df_daily = df_merged.groupby('region').resample('D').sum().reset_index()
    pivot_df = df_daily.pivot(index='DateTime', columns='region', values='Average Wind Speed (m/s)')
    pivot_df['TotalDailySumWindSpeed(m/s)_N'] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.rename(
                columns = {'N': 'DailySumWindSpeed(m/s)_N', 'NE': 'DailySumWindSpeed(m/s)_NE', 
                        'S': 'DailySumWindSpeed(m/s)_S', 'SE': 'DailySumWindSpeed(m/s)_SE'})
    #pivot_df['DateTime'] = pivot_df.index
    return pivot_df


def get_consumption_data_cleaned():
    df_consumption = gxdrive.read_file_from_xdrive_as_df("Consumption_Data.xlsx")
    df_consumption = df_consumption[['DateForecastCET','DateValueCET','VolumeMWh','PowerPriceAreaCode']]
    df_consumption.rename(columns={
        'DateForecastCET': 'DateTime',
        'DateValueCET': 'ForecastedDate',
        'VolumeMWh': 'Consumption'
    }, inplace=True)

    df_consumption['Region'] = df_consumption['PowerPriceAreaCode'].replace(dm.region_dict_power_price_area_code)
    df_consumption = df_consumption.drop(['PowerPriceAreaCode'], axis=1)
    df_agg = df_consumption.groupby(['DateTime', 'ForecastedDate', 'Region'], as_index=False)['Consumption'].sum()
    # Pivot the DataFrame to get separate columns for each region
    df_pivot = df_agg.pivot(index=['DateTime', 'ForecastedDate'], columns='Region', values='Consumption').reset_index()
    # Add a new column for aggregated values (sum of all regions)
    df_pivot['Total Consumption(MWh)'] = df_pivot[['N', 'NE', 'S', 'SE/CW']].sum(axis=1)
    df_pivot.columns.names = [None]
    df_pivot.rename(columns=lambda x: 'Consumption(MWh)_' + x if x in ['N', 'NE', 'S', 'SE/CW'] else x, inplace=True)

    return df_pivot

def get_wind_generation_data_cleaned():
    df_wind_2024 = gxdrive.read_file_from_xdrive_as_df('WindGeneration_Cleaned.csv')
    df_wind_2024['DateTime'] = pd.to_datetime(df_wind_2024['DateTime'])
    pivot_df = df_wind_2024.pivot(index='DateTime', columns='Region', values='Generation (MWavg)')
    pivot_df = pivot_df.rename(columns = {'N': 'Wind_Generation_N(MWavg)', 'NE': 'Wind_Generation_NE(MWavg)', 'S': 'Wind_Generation_S(MWavg)'})
    pivot_df['DateTime'] = pivot_df.index
    pivot_df = pivot_df.reset_index(drop = True)
    pivot_df['Wind_Generation_SUM(MWavg)'] = pivot_df[['Wind_Generation_N(MWavg)', 'Wind_Generation_NE(MWavg)', 'Wind_Generation_S(MWavg)']].sum(axis=1)

    return pivot_df


def merge_prices_and_consumption():

    df_prices = merge_market_and_settlement_price_merged()
    df_consumption = get_consumption_data_cleaned()
    df_all = pd.merge(df_prices, df_consumption, on='DateTime', how='inner')
    return df_all



if __name__=="__main__":
    get_data_set_and_split_to_train_and_test()
