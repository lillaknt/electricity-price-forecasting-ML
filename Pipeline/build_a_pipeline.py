import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from split_test_and_train import split_data_into_test_train_validation
from split_test_and_train import split_data_into_test_train_validation_for_P0_and_P1
from imputation import impute_missing_values
from feature_selection import feature_selection
from sklearn.preprocessing import PowerTransformer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
xdrive_path = os.path.join(parent_dir, 'xdrive')
sys.path.append(xdrive_path)
import get_files_from_xdrive as gxdrive

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
                                   drop_features= True, remove_outliers = False, 
                                   percentage_of_outliers = 0.01,
                                   train_size = 0.64, 
                                   val_size = 0.16,
                                   deal_with_skewness = False):
    """
    Args:
        target_columns(str); "P0" or "P1"
        imputation_strategy (str): Imputation strategy - "mean", "median", "most_frequent", "ffill", "bfill".
        scaling_strategy (str): Scaling method - "standard" or "minmax".
        features_to_drop (int): Number of least significant features to drop.

    """
    target_column = target_column
    data = gxdrive.read_file_from_xdrive_as_df("Prepared_Dataset_left_joint.csv")
    data = transform_to_P0_and_P1_columns(data)
    if remove_outliers:
        data = remove_top_x_percent_of_outliers(data, target_column, percentage_of_outliers)
    
    # Splitting into 64% train, 16% validation, and 20% test
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_into_test_train_validation_for_P0_and_P1(data, 
                                                                                                         target_column,
                                                                                                         train_size = train_size, 
                                                                                                         val_size = val_size)

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
    X_train = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train_imputed.columns, index=X_train_imputed.index)
    X_val = pd.DataFrame(scaler.transform(X_val_imputed), columns=X_val_imputed.columns, index=X_val_imputed.index)
    X_test = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_imputed.columns, index=X_test_imputed.index)

    # Feature selection
    if drop_features:
        #The feature selection keeeps dropping the P1 or P0 columns, so I made it optional
        X_train = feature_selection(X_train, y_train, features_to_drop=features_to_drop)
        X_val = X_val[X_train.columns]
        X_test = X_test[X_train.columns]

    
    if deal_with_skewness:
        X_train = deal_with_skewness_method(X_train)
        X_val = deal_with_skewness_method(X_val)
        X_test= deal_with_skewness_method(X_test)

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
    
    
def remove_top_x_percent_of_outliers(data, target_column, percentage = 0.01):
    # calculate the bounds
    lower_bound = data[target_column].quantile(percentage)
    upper_bound = data[target_column].quantile(1-percentage)
    #filter
    filtered_df = data[(data[target_column] > lower_bound) & (data[target_column] < upper_bound)]
    return filtered_df

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


def deal_with_skewness_method(dataset):
    pt = PowerTransformer(method="yeo-johnson")
    pt.fit(dataset)
    dataset  = pd.DataFrame(pt.transform(dataset + 0.00001), columns = dataset.columns) 
    return dataset
    

#build_pipeline_P0_foward_price(deal_with_skewness=True)