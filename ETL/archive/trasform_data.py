import pandas as pd
import glob
from io import BytesIO
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
etl_path = os.path.join(parent_dir, 'ETL')
eda_path = os.path.join(parent_dir, 'EDA')
sys.path.append(etl_path)
sys.path.append(eda_path)
import dictionary_mappings as dm
xdrive_path = os.path.join(parent_dir, 'xdrive')
# Add the xdrive path to sys.path
sys.path.append(xdrive_path)

# Now you can import your module from xdrive
import get_files_from_xdrive as gxdrive

#transform row files into cleaned data, save on xdrive, archive original file

#NOTE, the files were removed and archived, due to lack of storage

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
    

def clean_forward_price_and_save_csv():
   
    df_forward =  pd.read_excel(current_dir + '\\Forward_Price.xlsx')
    # Consistent column names across data sources
    df_forward.rename(columns={
        'ReferenceDateTimeOffset': 'DateTime',
        'DeliveryStartDateTimeOffset': 'Delivery_Start_Date_Forward_Price',
        'PriceMWh': 'Forward_Price_SE/CW(MWh)'
    }, inplace=True)

     # Consistent date format across data sources
    df_forward['DateTime'] = pd.to_datetime(df_forward['DateTime']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df_forward['Delivery_Start_Date_Forward_Price'] = pd.to_datetime(df_forward['Delivery_Start_Date_Forward_Price']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df_forward.drop(columns=['PriceI5MWh'], inplace=True)

    df_forward['DateTime'] = pd.to_datetime(df_forward['DateTime'])
    df_forward['Delivery_Start_Date_Forward_Price']=pd.to_datetime(df_forward['Delivery_Start_Date_Forward_Price'])

    # Save the resulting DataFrame to a CSV file
    df_forward.to_csv('Settlement_Price_Cleaned.csv', index=False)


def clean_settlement_price_and_save_csv():
   
    df_settlement =  pd.read_excel(current_dir + '\\Settlement_Price.xlsx')
    df_settlement = df_settlement[['DateValueCET', 'TimeValueCET', 'PowerPriceAreaCode', 'PriceMWh']]
    df_settlement['DateValueCET'] = pd.to_datetime(df_settlement['DateValueCET'])
    df_settlement['TimeValueCET'] = pd.to_timedelta(df_settlement['TimeValueCET'])
    df_settlement['DateTime'] = df_settlement['DateValueCET'] + df_settlement['TimeValueCET']
    df_settlement['DateTime'] = pd.to_datetime(df_settlement['DateTime'], format='%Y-%m-%d %H:%M:%S')
    df_filtered_settlement_price = df_settlement[df_settlement['PowerPriceAreaCode'] == 'BRAZIL_SOUTHEAST_CENTRALWEST']
    df_filtered_settlement_price['DateTime'] = df_settlement['DateTime'].dt.date

    df_aggregated_settlement_price = df_filtered_settlement_price.groupby('DateTime').agg(
        Average =('PriceMWh', 'mean'),
        STD =('PriceMWh', 'std'),
        MIN =('PriceMWh', 'min'),
        MAX =('PriceMWh', 'max'),
    ).reset_index()

    #remember to rename
    df_aggregated_settlement_price.rename(columns={
        'Average': 'Average_Settlement_Price_SE/CW(MWh)',
        'STD': 'Standard_Deviation_Settlement_Price_SE(MWh)',
        'MIN': 'Min_Settlement_Price_SE(MWh)',
        'MAX': 'Max_Settlement_Price_SE(MWh)'
    }, inplace=True)

    df_aggregated_settlement_price['DateTime'] = pd.to_datetime(df_aggregated_settlement_price['DateTime'])

    # Save the resulting DataFrame to a CSV file
    df_aggregated_settlement_price.to_csv('Settlement_Price_Cleaned.csv', index=False)



def clean_solar_generation_and_save_csv():
   
    df_solar =  pd.read_csv(current_dir + '\\Solar_Generation.csv')
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
    df_solar_filtered = df_solar[df_solar['DateTime'] >= '2021-01-01']
    df_aggregated_solar = df_solar_filtered.groupby(['DateTime', 'Region'], as_index=False)['Solar_Generated'].sum()
    # Pivot the DataFrame to get separate columns for each region
    df_pivot = df_aggregated_solar.pivot(index='DateTime', columns='Region', values='Solar_Generated').reset_index()
    # Add a new column for aggregated values (sum of all regions)
    df_pivot['Daily_Sum_Solar_Generated(MWavg)'] = df_pivot[['NE','SE/CW']].sum(axis=1)
    df_pivot.columns.names = [None]
    df_pivot.rename(columns=lambda x: 'Solar_Generated_' + x + '(MWavg)' if x in ['NE','SE/CW'] else x , inplace=True)
    df_pivot['DateTime'] = pd.to_datetime(df_pivot['DateTime'])

     # Save the resulting DataFrame to a CSV file
    df_pivot.to_csv('Solar_Generated_Cleaned.csv', index=False)


#if __name__=="__main__": df =  clean_solar_generation_and_save_csv()

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


if __name__=="__main__": 
    df = read_windgeneration_data_and_save_new_df()


