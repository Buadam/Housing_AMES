#!/usr/bin/env python
# coding: utf-8

# # Define helper functions for EDA
# Import modules
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def iscat(feature):
    """Decide whether a variable is categorical
    Input: DataFrame: df["feature"]
    Returns: True/False"""
    if (len(feature.unique())<20 or feature.dtype=='O'):
        return True
    else:
        return False

def sortcat(df, features):
    """Sort features into categorical and continuous groups
    Inputs: 
        df: DataFrame
        features: df columns: list of str
    Returns:
        cols_cat: list of columns identified as categorical
        cols_cont: list of columns identified as continuous
    """
    cols_cat=[]
    cols_cont=[]
    for feature in features:
        if iscat(df[feature]):
            cols_cat = np.append(cols_cat, feature)
        else:
            cols_cont = np.append(cols_cont, feature)
    return cols_cat, cols_cont


def analyzefeature(df_train, df_test, feature, feature2=None):
    """Analyze a single features or two features in the DataFrame
    Inputs: 
        df_train: DataFrame, training dataset
        df_test:  DataFrame, test dataset
        feature:  df column to analyze: str
        feature2: second df column to analyze: str
    Returns:
        None
    """
    plt.figure(figsize=(18, 6))
    housing = pd.concat([df_train, df_test], axis=0, sort=False)
    housing[[feature]].info()
    if iscat(df_train[feature]):
        print(feature + " is categorical")
        plt.subplot(1, 2, 1)
        y1 = (housing[feature].fillna('Missing')).value_counts(sort=True)
        x = np.arange(0, len(y1))
        sns.barplot(x, y1.values)
        plt.xticks(x, y1.index)
        plt.xlabel(feature)
        plt.ylabel("Total (train+test) counts")
        plt.subplot(1, 2, 2)
        sns.boxplot(x=feature, y="SalePrice", hue=feature2, data=df_train.fillna('Missing'), order=y1.index)
        plt.show()
        return True
    else:
        print(feature + " is continuous")
        plt.subplot(1, 3, 1)
        plt.hist(df_train[feature], bins=25)
        plt.subplot(1, 3, 2)
        x, y = ecdf(df_train[feature])
        plt.plot(x, y)
        plt.subplot(1, 3, 3)
        sns.scatterplot(x=feature, y="SalePrice", data=df_train)
        print(df_train[[feature, "SalePrice"]].corr())
        plt.show()
        return False



def ecdf(feature):
    """Calculate empirical cumulative distribution function
    Input: feature: DataFrame column: df["col"] 
    """
    x = np.arange(1, len(feature)+1)/len(feature)
    y = np.sort(feature)
    return x, y


def categorize(df_train, df_test, feature, bns=3, lbls=['1','2','3'], method="mean", fillna=False):
    """ Categorize the elements of a continuous feature into discrete categories.
    Inputs:
        df_train: DataFrame: training data: aggregated by "SalePrice"
        df_test:  DataFrame: test data: no "SalePrice" is available,
                   categorization is performed according to the training data.
        feature:  str: Specific column to categorize
        bns: integer: number of categories
        lbls: list of labels, num. elements equal to bns
        method: aggregation method. Default: "mean"
        fillna: replace missing values with most probable values: True/False
    Returns:
        feature_tr: transformed training set
        feature_tr_test: trasformed test set: "SalePrice"    
    """
    print(df_train.head())
    temp = df_train[[feature, "SalePrice"]].groupby(feature).agg(method)
    print(temp.head())
    tr, bins = pd.cut(temp["SalePrice"], bins=bns, labels=lbls, retbins=True)
    print(bins)
    print(tr.head())
    fna = tr.value_counts().idxmax()
    if fillna:
        feature_tr = df_train[feature].map(tr).fillna(fna).astype(str)
        feature_tr_test = df_test[feature].map(tr).fillna(fna).astype(str)
    else:
        feature_tr = df_train[feature].map(tr).astype(str)
        feature_tr_test = df_test[feature].map(tr).astype(str)
    print(feature_tr.head())
    return feature_tr, feature_tr_test


def safelog(array, zeromin=2):
    """Logarithm of dataset containing zero elements. 
    Inputs:
        array: numpy array
        zeromin: float
                 defines log(0) as the minimal value of log(nonzero elements) - zeromin
    """
    result = np.empty((len(array),1))
    b = (array!=0)
    result[b,0] = np.log(array[b])
    c = (array==0)
    result[c,0] = result[b,0].min()-zeromin
    return result[:,0]



