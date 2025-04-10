
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
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
#call files from xdrive, merge them together as needed

def merge_forward_and_settlement_price():
    df_forward_price = gxdrive.read_file_from_xdrive_as_df('Forward_Price_Cleaned.csv')
    df_settlemet_price = gxdrive.read_file_from_xdrive_as_df('Settlement_Price_Cleaned.csv')
    df_merged = pd.merge(df_forward_price, df_settlemet_price, on='DateTime', how='left')
    
    return df_merged

def merge_prices_and_hydro():

    df_prices = merge_forward_and_settlement_price()
    df_inflow = gxdrive.read_file_from_xdrive_as_df('HydroInflow_Cleaned.csv')
    
    df_merged = pd.merge(df_prices, df_inflow, on='DateTime', how='left')
    return df_merged

def merge_solar_with_prices_and_hydro():

    df_prices_and_hydro = merge_prices_and_hydro()
    df_solar = gxdrive.read_file_from_xdrive_as_df('Solar_Generated_Cleaned.csv')
    
    df_merged = pd.merge(df_prices_and_hydro, df_solar, on='DateTime', how='left')
    return df_merged


def merge_wind_generation_wind_df():
    df = merge_solar_with_prices_and_hydro()
    df_wind_generation = gxdrive.read_file_from_xdrive_as_df('WindGeneration_Cleaned.csv')
    df_merged = pd.merge(df, df_wind_generation, on='DateTime', how='left')
    return df_merged

def merge_capacity_installed_with_df():
    df = merge_wind_generation_wind_df()
    df_wind_generation = gxdrive.read_file_from_xdrive_as_df('CapacityInstalledDaily_Cleaned.csv')
    df_merged = pd.merge(df, df_wind_generation, on='DateTime', how='left')
    return df_merged

def merge_consumption_with_all():

    df_all = merge_capacity_installed_with_df()
    df_consumption = gxdrive.read_file_from_xdrive_as_df('Consumption_Cleaned.csv')
    df_merged = pd.merge(df_all, df_consumption, on='DateTime', how='left')
  

    # Save the resulting DataFrame to a CSV file
    df_merged.to_csv('Prepared_Dataset_left_joint.csv', index=False)

if __name__=="__main__": 
    df = merge_consumption_with_all()
