import pandas as pd
import pytz
import os
import sys

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
etl_path = os.path.join(parent_dir, 'ETL')
xdrive_path = os.path.join(parent_dir, 'xdrive')
sys.path.append(etl_path)
sys.path.append(xdrive_path)

# Import external modules
import dictionary_mappings as dm
import get_files_from_xdrive as gxdrive

#transform row files into cleaned data, save on xdrive, archive original file

#NOTE, the files were removed and archived, due to lack of storage

def clean_forward_price_and_save_csv():
    
    df_forward =  gxdrive.read_file_from_xdrive_as_df('Forward_Price.xlsx')
    
    df_forward.rename(columns={
        'ReferenceDateTimeOffset': 'DateTime',
        'DeliveryStartDateTimeOffset': 'Delivery_Start_Date_Forward_Price',
        'PriceMWh': 'Forward_Price_SE/CW(MWh)'
    }, inplace=True)
   # Consistent date format across data sources
    df_forward['DateTime'] = pd.to_datetime(df_forward['DateTime']).dt.strftime('%Y-%m-%d')
    df_forward['Delivery_Start_Date_Forward_Price'] = pd.to_datetime(df_forward['Delivery_Start_Date_Forward_Price']).dt.strftime('%Y-%m-%d')
    df_forward.drop(columns=['PriceI5MWh'], inplace=True)

    # Save the resulting DataFrame to a CSV file 
    df_forward.to_csv('Forward_Price_Cleaned.csv', index=False)
    
    return df_forward


def clean_settlement_price_and_save_csv():
    df_settlement = gxdrive.read_file_from_xdrive_as_df('Settlement_Price.xlsx')
    df_settlement = df_settlement[['DateValueCET', 'TimeValueCET', 'PowerPriceAreaCode', 'PriceMWh']]
    
    df_settlement['DateValueCET'] = pd.to_datetime(df_settlement['DateValueCET'])
    df_settlement['TimeValueCET'] = pd.to_timedelta(df_settlement['TimeValueCET'])
    
    df_settlement['DateTime'] = df_settlement['DateValueCET'] + df_settlement['TimeValueCET']
    
    # Convert to Brazilian local time (UTC-3 or UTC-2 for daylight saving time)
    cet = pytz.timezone('CET')  # Central European Time
    brazil = pytz.timezone('America/Sao_Paulo')  # Brazil time zone
    
    df_settlement['DateTime'] = df_settlement['DateTime'].dt.tz_localize(cet, ambiguous='NaT').dt.tz_convert(brazil)
    # Round to the date in Brazilian time zone for daily aggregation
    df_settlement['DateTime'] = df_settlement['DateTime'].dt.date
    
    df_filtered_settlement_price = df_settlement[df_settlement['PowerPriceAreaCode'] == 'BRAZIL_SOUTHEAST_CENTRALWEST']
    
    df_aggregated_settlement_price = df_filtered_settlement_price.groupby('DateTime').agg(
        Average=('PriceMWh', 'mean'),
        STD=('PriceMWh', 'std'),
        MIN=('PriceMWh', 'min'),
        MAX=('PriceMWh', 'max'),
    ).reset_index()
    
    df_aggregated_settlement_price.rename(columns={
        'Average': 'Average_Settlement_Price_SE/CW(MWh)',
        'STD': 'Standard_Deviation_Settlement_Price_SE(MWh)',
        'MIN': 'Min_Settlement_Price_SE(MWh)',
        'MAX': 'Max_Settlement_Price_SE(MWh)'
    }, inplace=True)
    
    df_aggregated_settlement_price['DateTime'] = pd.to_datetime(df_aggregated_settlement_price['DateTime'])
    
    df_aggregated_settlement_price.to_csv('Settlement_Price_Cleaned.csv', index=False)



def merge_all_wind_generation_files_into_one():

    file_paths = ['WindGeneration2021.csv', 'WindGeneration2022.csv', 'WindGeneration2023.csv', 'WindGeneration2024.csv'] 

    # Read and concatenate all CSV files
    df = pd.concat([pd.read_csv(etl_path +"\\"+ file) for file in file_paths], ignore_index=True)

    # Save the merged DataFrame to a new CSV file
    df.to_csv(etl_path +"\\"+ 'WindGeneration.csv', index=False)


def merge_weather_files_into_one():

    file_paths = ['Weather_2024.csv', 'Weather_2023.csv', 'Weather_2022.csv', 'Weather_2021.csv'] 

    # Read and concatenate all CSV files
    df = pd.concat([pd.read_csv(etl_path +"\\"+ file) for file in file_paths], ignore_index=True)

    # Save the merged DataFrame to a new CSV file
    df.to_csv(etl_path +"\\"+ 'WeatherData.csv', index=False)

 
def read_and_clean_inflow_data_and_save_new_df():
    df_hydro = gxdrive.read_file_from_xdrive_as_df("Hydro_Daily.xlsx")
    df_hydro = df_hydro[['DateValueCET', 'PowerPriceAreaCode', 'Value', 'Location']]
    df_hydro['Region'] = df_hydro['Location'].replace(dm.region_dict_bruno)
    df_hydro = df_hydro.rename(columns={'DateValueCET': 'DateTime', 'Value': 'Inflow'})
    df_hydro = df_hydro.drop(['PowerPriceAreaCode', 'Location'], axis=1)
    df_agg = df_hydro.groupby(['DateTime', 'Region'], as_index=False)['Inflow'].sum()
    # Pivot the DataFrame to get separate columns for each region
    df_pivot = df_agg.pivot(index='DateTime', columns='Region', values='Inflow').reset_index()
    # Add a new column for aggregated values (sum of all regions)
    df_pivot['Daily_Sum_Hydro_Inflow(MWavg)'] = df_pivot[['N', 'NE', 'S', 'SE/CW']].sum(axis=1)
    df_pivot.columns.names = [None]
    df_pivot.rename(columns=lambda x: 'Hydro_Inflow_' + x + '(MWavg)' if x in ['N', 'NE', 'S', 'SE/CW'] else x , inplace=True)
    #df_pivot = df_pivot.rename(columns = {'DateTime_(MWavg)': 'DateTime'})
    df_pivot['DateTime'] = pd.to_datetime(df_pivot['DateTime'])
    df_pivot = df_pivot[df_pivot['DateTime'] >= '2021-01-01']
    df_pivot.to_csv('HydroInflow_Cleaned.csv', index=False)
    

def clean_solar_generation_and_save_csv():
    df_solar = gxdrive.read_file_from_xdrive_as_df("Solar_Generation.csv")
    df_solar['Region'] = df_solar['id_subsistema'].replace(dm.region_dict_portugese)  
    df_solar.rename(columns={
        'din_instante': 'DateTime',
        'val_geraenergiaconmwmed': 'Solar_Generated'
    }, inplace=True)
    df_solar['DateTime'] = pd.to_datetime(df_solar['DateTime'])
    df_solar['Month'] = df_solar['DateTime'].dt.month_name()
    df_solar['Month'] = df_solar['Month'].map(dm.month_map)
    
    df_solar['DateTime'] = df_solar['DateTime'].dt.date
    df_solar['DateTime'] = pd.to_datetime(df_solar['DateTime'])
    
    # Filter data for dates from 2021-01-01 onwards
    df_solar_filtered = df_solar[df_solar['DateTime'] >= '2021-01-01']
    
    # Aggregate data by summing and averaging solar generation per region and date
    df_aggregated_solar = df_solar_filtered.groupby(['DateTime', 'Region'], as_index=False).agg(
        Solar_Sum=('Solar_Generated', 'sum'),
        Solar_Mean=('Solar_Generated', 'mean')
    )
    
    # Pivot the DataFrame to create separate columns for each region
    df_pivot_sum = df_aggregated_solar.pivot(index='DateTime', columns='Region', values='Solar_Sum').reset_index()
    df_pivot_mean = df_aggregated_solar.pivot(index='DateTime', columns='Region', values='Solar_Mean').reset_index()
    
    # Rename columns for clarity
    df_pivot_sum.columns.names = [None]
    df_pivot_mean.columns.names = [None]
    
    df_pivot_sum.rename(columns=lambda x: 'Solar_Generated_Sum_' + x + '(MWavg)' if x in ['NE', 'SE/CW'] else x, inplace=True)
    df_pivot_mean.rename(columns=lambda x: 'Solar_Generated_Mean_' + x + '(MWavg)' if x in ['NE', 'SE/CW'] else x, inplace=True)
    
    # Merge sum and mean DataFrames
    df_pivot = pd.merge(df_pivot_sum, df_pivot_mean, on='DateTime', how='outer')
    
    # Add a new column for daily aggregated sum (sum of all regions)
    df_pivot['Daily_Sum_Solar_Generated(MWavg)'] = df_pivot[['Solar_Generated_Sum_NE(MWavg)', 'Solar_Generated_Sum_SE/CW(MWavg)']].sum(axis=1)
    
    # Convert DateTime back to datetime format
    df_pivot['DateTime'] = pd.to_datetime(df_pivot['DateTime'])
    
    # Save the resulting DataFrame to a CSV file
    df_pivot.to_csv('Solar_Generated_Cleaned.csv', index=False)


def read_windgeneration_data_and_save_new_df():
   
    df =  pd.read_csv(etl_path +"\\"+ 'WindGeneration.csv')
    #choose relevant columns
    df = df[['Day of din_instante', 'Month of din_instante','id_subsistema','val_geraenergiaconmwmed']]
    df = df.rename(columns = {'Day of din_instante': 'Day', 
                         'Month of din_instante': 'Month',
                         'id_subsistema' : 'Region',
                         'val_geraenergiaconmwmed' : 'Generation (MWavg)'
                         })
    #unique regions = ['Nordeste', 'Norte', 'Sul']
    df['Region'] = df['Region'].replace(dm.region_dict_portugese)
    # Deal with Month being in 'August 2021' format
    df[['month', 'year']] = df['Month'].str.split(' ', expand=True)
    df['month'] = df['month'].map(dm.month_map)
    df['DateTime'] = pd.to_datetime(df[['year', 'month', 'Day']])
    df = df.drop(columns = ['Day', 'Month', 'month', 'year'])
    df['Generation (MWavg)'] = pd.to_numeric(df['Generation (MWavg)'], errors='coerce')
    df_grouped = df.groupby(['DateTime', 'Region'], as_index=False).agg({
    'Generation (MWavg)': 'sum'
})
    df_grouped['DateTime'] = pd.to_datetime(df_grouped['DateTime'])
    pivot_df = df_grouped.pivot(index='DateTime', columns='Region', values='Generation (MWavg)')
    pivot_df = pivot_df.rename(columns = {'N': 'Wind_Generation_N(MWavg)', 'NE': 'Wind_Generation_NE(MWavg)', 'S': 'Wind_Generation_S(MWavg)'})
    pivot_df['DateTime'] = pivot_df.index
    pivot_df = pivot_df.reset_index(drop = True)
    pivot_df['Wind_Generation_SUM(MWavg)'] = pivot_df[['Wind_Generation_N(MWavg)', 'Wind_Generation_NE(MWavg)', 'Wind_Generation_S(MWavg)']].sum(axis=1)
    # Save the resulting DataFrame to a CSV file
    pivot_df.to_csv('WindGeneration_Cleaned.csv', index=False)

    return pivot_df

def read_and_clean_capacity_data_and_save_new_df():
    #https://dados.ons.org.br/dataset/capacidade-geracao -- source
    df_capacity = gxdrive.read_file_from_xdrive_as_df("CAPACIDADE_GERACAO.csv")
    df = pd.read_csv(etl_path + "\\CAPACIDADE_GERACAO.csv", sep=';')
    # relevant columns: id_subsistema, dat_entradateste, dat_entradaoperacao, dat_sesativacao, val_potenciaefetiva
    df1 = df[['id_subsistema', 'dat_entradateste', 'dat_entradaoperacao', 'dat_desativacao',  'val_potenciaefetiva', 'nom_tipousina', 'nom_unidadegeradora']]
    df1_active = df1[df1['dat_desativacao'].isna()]
    df1['dat_entradaoperacao'] = pd.to_datetime(df1['dat_entradaoperacao'])
    # There were many new capacities installed throughout 2021 til 2024
    # this means I need to transform the df, to have each day and then sum up the capacity for that day
    #count_2022 = (df1['dat_entradaoperacao'].dt.year == 2022).sum()

    date_range = pd.date_range(start='2021-01-01', end='2024-10-31', freq='D')
    date_df = pd.DataFrame(date_range, columns=['datetime'])
    df1['dat_entradaoperacao'] = pd.to_datetime(df1['dat_entradaoperacao'])
    df1['dat_desativacao'] = pd.to_datetime(df1['dat_desativacao'])
    df1 = df1[df1['id_subsistema'] != 'PY'] #unknown region, dropping this
           # Apply the mappings to 'nom_tipousina' (energy type) and 'id_subsistema' (region)
    df1['nom_tipousina'] = df1['nom_tipousina'].replace(dm.energy_type_mapping_capacity)
    df1['id_subsistema'] = df1['id_subsistema'].replace(dm.region_mapping_capacity)
    energy_types = df1['nom_tipousina'].unique()
    regions = df1['id_subsistema'].unique()
        # Dynamically create columns in date_df for each combination of energy type and region
    for energy_type in energy_types:
        for region in regions:
            col_name = f"{energy_type}_capacity_{region}(MWavg)"
            date_df[col_name] = 0  # Initialize with zero

    # For each date in the date range, calculate cumulative installed capacity
    # Calculate cumulative installed capacity for each day
    for i, current_date in enumerate(date_df['datetime']):
        for energy_type in energy_types:
            for region in regions:
                # Filter rows for the current date, energy type, and region
                active_capacities = df1[
                    (df1['dat_entradaoperacao'] <= current_date) &
                    ((df1['dat_desativacao'].isna()) | (df1['dat_desativacao'] > current_date)) &
                    (df1['id_subsistema'] == region) &
                    (df1['nom_tipousina'] == energy_type)
                ]['val_potenciaefetiva'].sum()
                
                # Update the corresponding column in date_df
                col_name = f"{energy_type}_capacity_{region}(MWavg)"
                date_df.loc[i, col_name] = active_capacities

    date_df = date_df.loc[:, (date_df != 0).any(axis=0)]
    # List of regions
    regions = ['NE', 'N', 'SE/CW', 'S']

    # Loop through each region and sum the capacities for that region
    for region in regions:
        # Filter columns that end with the region name and '(MWavg)'
        region_columns = [col for col in date_df.columns if col.endswith(f"{region}(MWavg)")]
        
        # Create a new column for the region's total daily capacity
        date_df[f'Total_capacity_{region}(MWavg)'] = date_df[region_columns].sum(axis=1)
    date_df = date_df.rename(columns = {'datetime': 'DateTime'})

     # Save the resulting DataFrame to a CSV file
    date_df.to_csv('CapacityInstalledDaily_Cleaned.csv', index=False)


def clean_consumption_data_and_save_csv():
    df_consumption = pd.read_excel(current_dir + '\\Consumption_Data.xlsx')
    df_consumption = df_consumption[['DateForecastCET','DateValueCET','VolumeMWh','PowerPriceAreaCode', 'ForecastRangeCET' ]]
    df_consumption.rename(columns={
        'DateForecastCET': 'DateTime',
        'DateValueCET': 'ForecastedDate',
        'VolumeMWh': 'Consumption'
    }, inplace=True)

    df_consumption['Region'] = df_consumption['PowerPriceAreaCode'].replace(dm.region_dict_power_price_area_code)
    consumption_df = df_consumption[(df_consumption['DateTime'] >= '2021-01-01') & 
                                (df_consumption['ForecastedDate'] >= '2021-01-01')]
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
    df_pivot['Daily_Sum_Consumption(MWh)'] = df_pivot[['N', 'NE', 'S', 'SE/CW']].sum(axis=1)
    df_pivot.columns.names = [None]
    df_pivot.rename(columns=lambda x: 'Consumption(MWh)_' + x if x in ['N', 'NE', 'S', 'SE/CW'] else x, inplace=True)

    # Save the resulting DataFrame to a CSV file
    df_pivot.to_csv('Consumption_Cleaned.csv', index=False)
    
    
if __name__=="__main__": 
    df = clean_forward_price_and_save_csv()
    



