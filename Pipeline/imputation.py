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
