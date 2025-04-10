import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


# Get the current directory of the file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up two levels to the grandparent directory
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))

# Build the path to the 'xdrive' folder inside the grandparent directory
xdrive_path = os.path.join(grandparent_dir, 'xdrive')
sys.path.append(xdrive_path)
import get_files_from_xdrive as gxdrive


import pandas as pd
from sklearn.impute import SimpleImputer

import pandas as pd
from sklearn.impute import SimpleImputer

def impute_missing_values(X, strategy="mean"):

    if strategy == "ffill":
        return X.ffill()
    elif strategy == "bfill":
        return X.bfill()
    elif strategy in ["mean", "median", "most_frequent"]:
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy=strategy)
        return pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    else:
        raise ValueError(f"Invalid imputation strategy: {strategy}")
    

def feature_selection(X_dataset, y_dataset, features_to_drop):
    """
    Drops the specified number of least significant features from the dataset using Variance Threshold 
    and Recursive Feature Elimination (RFE).

    Args:
        X_dataset (pd.DataFrame): Feature dataset.
        y_dataset (pd.Series): Target variable.
        features_to_drop (int): Number of features to drop based on significance.

    Returns:
        pd.DataFrame: Dataset with the least significant features removed.
    """

    # Step 1: Variance Threshold
    sel = VarianceThreshold(threshold=(0.99 * (1 - 0.99)))  # High variance threshold
    X_reduced = sel.fit_transform(X_dataset)

    # Mask of selected features after Variance Threshold
    mask_variance = sel.get_support()
    X_reduced_df = pd.DataFrame(X_reduced, columns=X_dataset.columns[mask_variance], index=X_dataset.index)

    # Step 2: Recursive Feature Elimination (RFE)
    estimator = LinearRegression()
    total_features = X_reduced_df.shape[1]
    features_to_select = total_features - features_to_drop  # Calculate features to retain
    selector = RFE(estimator, n_features_to_select=features_to_select)
    selector.fit(X_reduced_df, y_dataset)

    # Mask of selected features after RFE
    mask_rfe = selector.get_support()
    X_selected_df = X_reduced_df.loc[:, mask_rfe]
    return X_selected_df

def build_pipeline(imputation_strategy="mean", scaling_strategy="standard", features_to_drop=5):
    """
    Args:
        imputation_strategy (str): Imputation strategy - "mean", "median", "most_frequent", "ffill", "bfill".
        scaling_strategy (str): Scaling method - "standard" or "minmax".
        features_to_drop (int): Number of least significant features to drop.
    """
    data = gxdrive.read_file_from_xdrive_as_df("Prepared_Dataset_left_joint.csv")
    target_column = "Forward_Price_SE/CW(MWh)"

    # Splitting into 64% train, 16% validation, and 20% test
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_into_test_train_validation(data, target_column)

 # Impute missing values for each dataset
    X_train_imputed = impute_missing_values(X_train, strategy=imputation_strategy)
    X_val_imputed = impute_missing_values(X_val, strategy=imputation_strategy)
    X_test_imputed = impute_missing_values(X_test, strategy=imputation_strategy)

    #Scaling strategies defined
    if scaling_strategy == "standard":
        scaler = StandardScaler()
    elif scaling_strategy == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaling_strategy. Choose 'standard' or 'minmax'.")

    #Scale training data and transform all datasets
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train_imputed.columns, index=X_train_imputed.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val_imputed), columns=X_val_imputed.columns, index=X_val_imputed.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_imputed.columns, index=X_test_imputed.index)


        # Feature selection
    X_train_selected = feature_selection(X_train_scaled, y_train, features_to_drop=features_to_drop)
    X_val_selected = X_val_scaled[X_train_selected.columns]
    X_test_selected = X_test_scaled[X_train_selected.columns]

    return X_train_selected, X_val_selected, X_test_selected, y_train, y_val, y_test



def build_pipeline_P0_foward_price(target_column = "P0", imputation_strategy="mean", 
                                   scaling_strategy="standard", features_to_drop=5,
                                   drop_features= True):
    """
    Args:
        target_columns(str); "P0" or "P1"
        imputation_strategy (str): Imputation strategy - "mean", "median", "most_frequent", "ffill", "bfill".
        scaling_strategy (str): Scaling method - "standard" or "minmax".
        features_to_drop (int): Number of least significant features to drop.
    """
    data = gxdrive.read_file_from_xdrive_as_df("Prepared_Dataset_left_joint.csv")
    data = transform_to_P0_and_P1_columns(data)
    target_column = target_column

    # Splitting into 64% train, 16% validation, and 20% test
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_into_test_train_validation_for_P0_and_P1(data, target_column)

 # Impute missing values for each dataset
    X_train_imputed = impute_missing_values(X_train, strategy=imputation_strategy)
    X_val_imputed = impute_missing_values(X_val, strategy=imputation_strategy)
    X_test_imputed = impute_missing_values(X_test, strategy=imputation_strategy)

    #Scaling strategies defined
    if scaling_strategy == "standard":
        scaler = StandardScaler()
    elif scaling_strategy == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaling_strategy. Choose 'standard' or 'minmax'.")

    #Scale training data and transform all datasets
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train_imputed.columns, index=X_train_imputed.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val_imputed), columns=X_val_imputed.columns, index=X_val_imputed.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_imputed.columns, index=X_test_imputed.index)

    # Feature selection
    if drop_features:
        #The feature selection keeeps dropping the P1 or P0 columns, so I made it optional
        X_train_selected = feature_selection(X_train_scaled, y_train, features_to_drop=features_to_drop)
        X_val_selected = X_val_scaled[X_train_selected.columns]
        X_test_selected = X_test_scaled[X_train_selected.columns]
        return X_train_selected, X_val_selected, X_test_selected, y_train, y_val, y_test
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

def split_data_into_test_train_validation_for_P0_and_P1(data, target_column):

    # Sort data by DateTime to preserve time series order
    data = data.sort_index()
    # Split into features (X) and target (y)
    X = data.drop(columns=[target_column]) 
    y = data[target_column]  

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, train_size=0.8, random_state=42) #(80-20% split)
    # Perform the second split (train vs. validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, train_size=0.8, random_state=42)  #(further split into train (80% and validation 20%))
    return X_train, X_val, X_test, y_train, y_val, y_test

def make_basic_features_pipeline():
    data = gxdrive.read_file_from_xdrive_as_df("Prepared_Dataset_left_joint.csv")
    target_column = "Forward_Price_SE/CW(MWh)"
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_into_test_train_validation(data, target_column)
    #scaling happens in feature selection
    #There is some randomness going on the test dataset and validation, 
    #We will make the train to decise which features to drop and then look at the rest
    X_train = feature_selection(X_train, y_train,  False)
    X_test  = transform_test_and_val_data(X_train, X_test)
    X_val = transform_test_and_val_data(X_train, X_val)

    X_train = X_train[['DateTime', 'Delivery_Start_Date_Forward_Price',
        'Average_Settlement_Price_SE/CW(MWh)',  'Hydro_Inflow_N(MWavg)', 
        'Hydro_Inflow_NE(MWavg)',
       'Hydro_Inflow_S(MWavg)', 'Hydro_Inflow_SE/CW(MWavg)',
       'Daily_Sum_Hydro_Inflow(MWavg)']]
    X_val = X_val[['DateTime', 'Delivery_Start_Date_Forward_Price',
        'Average_Settlement_Price_SE/CW(MWh)',  'Hydro_Inflow_N(MWavg)', 
        'Hydro_Inflow_NE(MWavg)',
       'Hydro_Inflow_S(MWavg)', 'Hydro_Inflow_SE/CW(MWavg)',
       'Daily_Sum_Hydro_Inflow(MWavg)']]
    X_test = X_test[['DateTime', 'Delivery_Start_Date_Forward_Price',
        'Average_Settlement_Price_SE/CW(MWh)',  'Hydro_Inflow_N(MWavg)', 
        'Hydro_Inflow_NE(MWavg)',
       'Hydro_Inflow_S(MWavg)', 'Hydro_Inflow_SE/CW(MWavg)',
       'Daily_Sum_Hydro_Inflow(MWavg)']]

    return X_test,y_test, X_train, X_val, y_train, y_val

def make_basic_features_pipeline_for_settlement_price_prediction():
    data = gxdrive.read_file_from_xdrive_as_df("Prepared_Dataset_left_joint.csv")
    target_column = "Average_Settlement_Price_SE/CW(MWh)"
    data = data.dropna(subset=["Average_Settlement_Price_SE/CW(MWh)"])
    data = data[['Average_Settlement_Price_SE/CW(MWh)', 'DateTime',   'Hydro_Inflow_N(MWavg)', 
        'Hydro_Inflow_NE(MWavg)','Hydro_Inflow_S(MWavg)', 'Hydro_Inflow_SE/CW(MWavg)',
       'Daily_Sum_Hydro_Inflow(MWavg)']]
    # Select rows with unique DateTime values
    data  =  data.drop_duplicates(subset='DateTime')
    data.index = data['DateTime']
    data = data.drop(columns = 'DateTime')
    data = remove_outliers(data)

    # Splitting into 64% train, 16% validation, and 20% test
    X = data.drop(columns=[target_column]) 
    y = data[target_column]   
    #X['DateTime'] = (X['DateTime'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, train_size=0.8, random_state=42) #(80-20% split)
    # Perform the second split (train vs. validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, train_size=0.8, random_state=42)  #(further split into train (80% and validation 20%))

    imputation_strategy = "mean"
    # Impute missing values for each dataset
    X_train_imputed = impute_missing_values(X_train, strategy=imputation_strategy)
    X_val_imputed = impute_missing_values(X_val, strategy=imputation_strategy)
    X_test_imputed = impute_missing_values(X_test, strategy=imputation_strategy)

    scaler = StandardScaler()

    #Scale training data and transform all datasets
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train_imputed.columns, index=X_train_imputed.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val_imputed), columns=X_val_imputed.columns, index=X_val_imputed.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_imputed.columns, index=X_test_imputed.index)


    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test




def remove_outliers(data):
        # Sort data by DateTime to preserve time series order
    if 'DateTime' in data.columns:
        data['DateTime'] = pd.to_datetime(data['DateTime'])
        data = data.sort_values('DateTime')

    # Select only numeric and datetime columns
    data = data.select_dtypes(include=['number', 'datetime64'])

    # Calculate the first (Q1) and third (Q3) quartiles for each numeric column
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)

    # Calculate IQR (Interquartile Range)
    IQR = Q3 - Q1

    # Filter out the outliers based on the IQR
    data_filtered = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

    return data_filtered



def transform_test_and_val_data(X_train, X_toapply):
    X_toapply = X_toapply[X_train.columns]
    scaler = StandardScaler()
    X_toapply = pd.DataFrame(scaler.fit_transform(X_toapply), 
                                  columns=X_toapply.columns, index=X_toapply.index)
    X_toapply  = pd.DataFrame(X_toapply, columns=X_train.columns)
    return X_toapply


def transform_to_P0_and_P1_columns(data):
    import pandas as pd
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data['Delivery_Start_Date_Forward_Price'] = pd.to_datetime(data['Delivery_Start_Date_Forward_Price'])
    #correct
    data['DateTime_YearMonth'] = data['DateTime'].dt.to_period('M')
    #correct
    data['DateTime_1month_ahead'] = (data['DateTime'] + pd.DateOffset(months=1)).dt.to_period('M')
    #forward price period
    data['PriceDate_YearMonth'] = data['Delivery_Start_Date_Forward_Price'].dt.to_period('M')

    df_p0 = data[['DateTime', 'DateTime_YearMonth', 'PriceDate_YearMonth', 'Forward_Price_SE/CW(MWh)']].copy()
    df_p0 = df_p0[df_p0['PriceDate_YearMonth'] == df_p0['DateTime_YearMonth']]
    df_p0 = df_p0.rename(columns={'Forward_Price_SE/CW(MWh)': 'P0'})

    df_p1 = data[['DateTime', 'DateTime_YearMonth', 'PriceDate_YearMonth', 'DateTime_1month_ahead', 'Forward_Price_SE/CW(MWh)']].copy()
    df_p1 = df_p1[df_p1['PriceDate_YearMonth'] == df_p1['DateTime_1month_ahead']]
    df_p1 = df_p1.rename(columns={'Forward_Price_SE/CW(MWh)': 'P1'})

    merged_df = pd.merge(df_p0[['DateTime', 'P0']], 
                        df_p1[['DateTime', 'P1']], 
                        on='DateTime', 
                        how='outer')
    data = data.drop_duplicates(subset='DateTime')
    final_df = pd.merge(merged_df, 
                    data, 
                    on='DateTime', 
                    how='left')

    final_df= final_df.set_index('DateTime')  # Set DateTime as index
    # Step 3: Drop the unnecessary columns
    final_df.drop(columns=['Delivery_Start_Date_Forward_Price', 
                       'DateTime_YearMonth', 'Forward_Price_SE/CW(MWh)',
                       'DateTime_1month_ahead', 'PriceDate_YearMonth' ], inplace=True)

    #TODO This is bad practice, but just trying to see how it performs
    final_df['P0'] = final_df['P0'].fillna(final_df['P1'])

    return final_df



def split_data_into_test_train_validation(data, target_column):

    # Sort data by DateTime to preserve time series order
    if 'DateTime' in data.columns:
        data['DateTime'] = pd.to_datetime(data['DateTime'])
        data = data.sort_values('DateTime')

    # Split into features (X) and target (y)
    X = data.drop(columns=[target_column]) 
    y = data[target_column]  

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



