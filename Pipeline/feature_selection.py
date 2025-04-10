import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE



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

