"""Check correlation between variables"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot as plt

# Read data
housing_train = pd.read_csv('train.csv', sep=',', index_col='Id')
housing_test = pd.read_csv('test.csv', sep=',', index_col='Id')

features = housing_train.columns


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
        if iscat(data[feature]):
            cols_cat = np.append(cols_cat, feature)
        else:
            cols_cont = np.append(cols_cont, feature)
    return cols_cat, cols_cont


def analyzefeature(feature, feature2=None):
    if iscat(housing_train[feature]):
        plt.subplot(1, 2, 1)
        y1 = (housing[feature].fillna('Missing')).value_counts(sort=True)
        x = np.arange(0, len(y1))
        sns.barplot(x, y1.values)
        plt.xticks(x, y1.index)
        plt.xlabel(feature)
        plt.ylabel("Total (train+test) counts")
        plt.subplot(1, 2, 2)
        sns.boxplot(x=feature, y="SalePrice", hue=feature2, data=housing_train.fillna('Missing'), order=y1.index)
        return True
    else:
        plt.subplot(1, 3, 1)
        plt.hist(housing_train[feature], bins=20)
        plt.subplot(1, 3, 2)
        x, y = ecdf(housing_train[feature])
        plt.plot(x, y)
        plt.subplot(1, 3, 3)
        sns.scatterplot(x=feature, y="SalePrice", data=housing_train)
        print(housing_train[[feature, "SalePrice"]].corr())
        return False


def ecdf(feature):
    x = np.arange(1, len(feature)+1)/len(feature)
    y = np.sort(feature)
    return x, y


# Sort categorical and continuous features
cols_cat, cols_cont = sortcat(housing_train, features)
# Treat the train and test data the same way
housing = pd.concat([housing_train, housing_test], axis=0, sort=False)

# ====================================== Select and transform features =================================================
features_to_drop = ["MSSubClass", "MSZoning", "Street", "YearRemodAdd", "GarageYrBlt", "MiscVal", "SalePrice"]
features_to_normalize = ["LotFrontage"]
features_to_dummies = ["MSZoning_tr", "Alley"]

# MSSubClass : correlated with HouseStyle and Age. Drop.
# MSZoning: not much difference between RH and RM, there is only a few RH -> collect them to RMH. Drop original feature
tr = {"RL": 'RL', "RM": 'RMH', "RH": 'RMH', "FV": "FV", "C (all)": "C"}
housing_train["MSZoning_tr"] = housing_train["MSZoning"].map(tr)
housing_test["MSZoning_tr"] = housing_test["MSZoning"].map(tr)

# LotFrontage: positive correlation with SalePrice (0.35). Keep feature and (log?) normalize
housing_train["LogLotFrontage"] = np.log(housing_train["LotFrontage"])
# Lot Area:


# Street and Alley: no significant correlation between the two, yet they have significant impact on price. However
# Street is mostly paved, and only 10 are GRVL -> Drop street
# Alley is mostly missing: 2720-120-80 ratio: Keep as missing
plt.figure()
analyzefeature("LogLotFrontage")
plt.show()

