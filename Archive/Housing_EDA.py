import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# Read data
housing_train = pd.read_csv('train.csv', sep=',', index_col='Id')
housing_train.head()
features = housing_train.columns


def iscat(feature):
    """Decide whether a variable is categorical"""
    print(feature.unique())
    if (len(feature.unique())<13 or feature.dtype=='O'):
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


n = len(features)
cols_cat, cols_cont = sortcat(housing_train, features)
cols_cont
cols_cat
n_cat = len(cols_cat)
n_cat
cols_cat[1]
plt.figure()
for i in range (1, 20):
    plt.subplot(4,5,i)
    sns.boxplot(x=cols_cat[i-1], y='SalePrice', data=housing_train)
plt.show()

plt.figure()
for i in range (20, 40):
    plt.subplot(4,5,i%20+1)
    sns.boxplot(x=cols_cat[i-1], y='SalePrice', data=housing_train)
plt.show()

plt.figure()
for i in range (40, n_cat+1):
    plt.subplot(4,5,i%20+1)
    sns.boxplot(x=cols_cat[i-1], y='SalePrice', data=housing_train)
plt.show()

plt.figure()
pd.plotting.scatter_matrix(housing_train[cols_cont], diagonal='kde')
plt.show()

plt.figure()
corr = housing_train[cols_cont].corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()


iscat(housing_train["YrSold"])
housing_train.dtypes
housing_train.info()
cols_cont
