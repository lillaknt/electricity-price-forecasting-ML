from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit()
# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
xdrive_path = os.path.join(parent_dir, 'xdrive')
sys.path.append(xdrive_path)

import get_files_from_xdrive as gxdrive

def split_data_into_test_train_validation(data, target_column):

    # Sort data by DateTime to preserve time series order
    if 'DateTime' in data.columns:
        data['DateTime'] = pd.to_datetime(data['DateTime'])
        data = data.sort_values('DateTime')

    # Split into features (X) and target (y)
    X = data.drop(columns=[target_column]) 
    y = data[target_column]  

    # # Convert datetime columns to numeric timestamps for modeling
    # for col in ['DateTime', 'Delivery_Start_Date_Forward_Price']:
    #     if col in X.columns:
    #         X[col] = pd.to_datetime(X[col]).astype(int) // 10**9  # Convert to seconds since epoch

    # Deal with Datetime and convert it to a numeric value
    X['DateTime'] = (X['DateTime'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
    X['Delivery_Start_Date_Forward_Price'] = pd.to_datetime(X['Delivery_Start_Date_Forward_Price'] )
    X['Delivery_Start_Date_Forward_Price'] = (X['Delivery_Start_Date_Forward_Price'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, train_size=0.8, random_state=42) #(80-20% split)

    # Perform the second split (train vs. validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, train_size=0.8, random_state=42)  #(further split into train (80% and validation 20%))

    return X_train, X_val, X_test, y_train, y_val, y_test



def split_data_into_test_train_validation_for_P0_and_P1(data, target_column, train_size, val_size):

    # Sort data by DateTime to preserve time series order
    data = data.sort_index()
   
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_into_test_train_validation_for_P0_and_P1_timeserries(data,target_column, train_size, val_size)

    return X_train, X_val, X_test, y_train, y_val, y_test


def split_data_into_test_train_validation_for_P0_and_P1_timeserries(data, target_column, 
                                                                    train_size = 0.64, 
                                                                    val_size = 0.16):

    # Sort data by DateTime to preserve time series order
    data = data.sort_index()
    # Calculate the indices to split the data
    train_end = int(len(data) * train_size)
    val_end = train_end + int(len(data) * val_size)
    # Create the splits
    train_data = data[:train_end] 
    val_data = data[train_end:val_end] 
    test_data = data[val_end:]  

    y_train = train_data[target_column]
    y_test = test_data[target_column]
    y_val = val_data[target_column]

    X_train = train_data.drop(columns = target_column)
    X_val = val_data.drop(columns = target_column)
    X_test = test_data.drop(columns = target_column)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
   
   
    """ 
    # For debug
    data = gxdrive.read_file_from_xdrive_as_df("Prepared_Dataset_left_joint.csv")
    target_column = "Forward_Price_SE/CW(MWh)"

    X_train, X_val, X_test, y_train, y_val, y_test = split_data_into_test_train_validation(data, target_column)

    print("Training set size:", X_train.shape)
    print("Validation set size:", X_val.shape)
    print("Test set size:", X_test.shape) 
    
    """


