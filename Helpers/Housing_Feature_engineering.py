import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot as plt

# Read data
housing_train = pd.read_csv('train.csv', sep=',', index_col='Id')
housing_test = pd.read_csv('test.csv', sep=',', index_col='Id')
housing_train.info()
housing_test.info()
features = housing_train.columns
train_rows = len(housing_train)
test_rows = len(housing_test)

def iscat(feature):
    """Decide whether a variable is categorical"""
    if (len(feature.unique())<20 or feature.dtype=='O'):
        return True
    else:
        return False


def sortcat(data, features):
    """Sort features into categorical and continuous groups"""
    cols_cat=[]
    cols_cont=[]
    for feature in features:
        print(feature)
        if iscat(data[feature]):
            cols_cat = np.append(cols_cat, feature)
        else:
            cols_cont = np.append(cols_cont, feature)
    return cols_cat, cols_cont


def featureselect(df, cols_to_dummies, cols_to_drop, cols_to_impute, train_rows):
    """Feature selection for train and test data
        input:
        df: DataFrame,
        cols_to_drop: list of column names
        cols_to_dummies: list of column names to convert to dummies
        output: DataFrame after feature selection
    """
    # Convert building/renovation year to age
    df["Age"] = df["YrSold"] - df["YearBuilt"]
    df["RenAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["GarAge"] = df["YrSold"] - df["GarageYrBlt"]

    # Convert dummy features to string
    df[cols_to_dummies] = df[cols_to_dummies].apply(lambda row: row.astype(str))
    # Get dummies for specific features
    dummies = pd.get_dummies(df[cols_to_dummies])
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(cols_to_dummies, axis=1)

    # Impute missing values with mean
    df[cols_to_impute].fillna(df[cols_to_impute].mean(), inplace=True)

    # Standardize continuous features
    sc = StandardScaler()
    df[cols_cont] = sc.fit_transform(df[cols_cont])

    # Drop unnecessary features
    print(df.columns)
    df = df.drop(cols_to_drop, axis=1)

    # Print info on remaining columns
    print('Columns after feature selection:\n')
    print(df.info())
    # Separate training and test data
    df_train = df.iloc[:train_rows+1, :]
    df_test = df.iloc[train_rows+1:, :]
    return df_train, df_test


# Sort categorical and continuous features
cols_cat, cols_cont = sortcat(housing_train, features)

cols_to_drop = ["YearBuilt", "YearRemodAdd", "GarageYrBlt", "MiscVal", "SalePrice"]
cols_to_dummies = cols_cat
cols_to_impute = cols_cont
df = pd.concat([housing_train, housing_test], axis=0, sort=False)
X_train_df, X_test_df = featureselect(df, cols_to_dummies, cols_to_drop, cols_to_impute, train_rows)

# ======================================== Inspect variable correlations ============================================== #

corr = X_train_df.corr()
sns.heatmap(corr[corr.abs() > 0.75],
            xticklabels=corr[corr.abs() > 0.75].columns.values,
            yticklabels=corr[corr.abs() > 0.75].columns.values)

for feature in corr.index:
    print(feature.value + "correlates with: ")
    print(list(zip({"feature": corr[feature][corr.abs() > 0.75].name, "correlation": corr[feature][corr.abs() > 0.75]})))

# Subtract the value of misc, keep only the category to account for the added value
# (remember to add misc value to the prediction)
y_train_df = housing_train["SalePrice"]-housing_train["MiscVal"]



# TODO: add misc value to the prediction