from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns


from Tools import encode_labels

max_nan_percentage = 0.15
seed = 7
np.random.seed(seed)

"""
Steps:
 understend witch feature more effective by LassoCV

 transform skewed data by log(1 + x)

 fit 0 or None by logic
 transform categorical number to categorical
 transform categorical by LabelEncoder
 transform categorical by dummies
 use ensemble methods 
 Out liars 
 if feature has > 15% missing data delete it
"""


def select_features(data, price):
    lr = LinearRegression()
    selector = RFE(lr, n_features_to_select=3)
    # selector = RFE(lr)
    selector.fit(data, price)

    print(selector.support_)
    print(selector.ranking_)

    column_i = []
    for i in range(len(selector.support_)):
        if ~selector.support_[i]:
            column_i.append(i)

    return column_i


def prepare_data():
    df = pd.read_csv(filepath_or_buffer="resources/train.csv").fillna(0)

    price = df['SalePrice'].as_matrix()

    df = df.drop(['SalePrice', 'Id'], 1)
    encode_labels(df)

    extract_columns = select_features(df.as_matrix(), price)

    print(len(df.columns.values))
    df = df.drop(df.columns[extract_columns], axis=1)
    print(len(df.columns.values))

    data = df.as_matrix()

    feature = [np.array(v) for v in data]
    label = list(map(lambda v: np.array([v]), price))

    return train_test_split(feature, label, test_size=0.3, random_state=seed)


def main():
    train_feature, test_feature, price, test_price = prepare_data()

    lr = LinearRegression()
    lr.fit(train_feature, price)
    print('coef_', lr.coef_)
    print('intercept_', lr.intercept_)

    pred = lr.predict(test_feature)

    # accuracy = r2_score(test_price, pred)
    accuracy = sqrt(mean_squared_error(test_price, pred))
    print('accuracy', accuracy)

    # plt.scatter(train_feature, price, color='b', marker='.')
    # plt.scatter(test_feature, test_price, color='r', marker='.')
    # plt.plot(test_feature, pred)
    # plt.show()


def replace_dummies(df):
    new_df = df
    for column_name in df:
        column = df[column_name]
        if column.dtype == 'object' or column.dtype == 'str':
            print('column_name', column_name)
            dummies = pd.get_dummies(column, prefix=column_name)
            print(dummies)
            new_df = new_df.drop([column_name], axis=1)
            # new_df = new_df.join(dummies)
            new_df = pd.concat([new_df, dummies], axis=1)

    return new_df


def show_zoomed_heatmap(df):
    corrmat = df.corr()

    k = 20  # number of variables for heatmap
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    print('cols', cols)

    cm = np.corrcoef(df[cols].values.T)
    print(cm)

    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                     xticklabels=cols.values)
    plt.show()


def show_multi_plot(df):
    sns.set()
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    sns.pairplot(df[cols], size=2)
    plt.show()


def show_heatmap(df):
    print(df.columns.values)

    corr = df.corr()

    print(corr.values.shape)
    print(corr.columns.values)
    corr_data = corr.values
    columns = corr.columns.values

    fig, ax = plt.subplots()
    im = ax.imshow(corr_data)

    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))

    ax.set_xticklabels(columns)
    ax.set_yticklabels(columns)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # for i in range(len(columns)):
    #     for j in range(len(columns)):
    #         text = ax.text(j, i, corr_data[i, j],
    #                        ha="center", va="center", color="w")

    fig.tight_layout()
    plt.show()


def get_percentage_missing(series):
    num = series.isnull().sum()
    den = len(series)
    return round(num / den, 2)


def drop_features_with_nan(df):
    nan_percentage = {}

    print(df.shape)

    for column_name in df:
        nan_percentage[column_name] = get_percentage_missing(df[column_name])

    column_to_drop = list(map(lambda x: x[0],
                              filter(lambda x: x[1] > max_nan_percentage, nan_percentage.items()))
                          )

    print(len(column_to_drop))
    print(column_to_drop)

    df.drop(labels=column_to_drop, axis=1, inplace=True)

    print(df.shape)


def drop_by_corr(df):
    culumns = ['GarageArea', 'TotRmsAbvGrd', '1stFlrSF']
    df.drop(labels=culumns, axis=1, inplace=True)


if __name__ == "__main__":
    # main()

    # show_multi_plot()

    df = pd.read_csv(filepath_or_buffer="resources/train.csv")
    df.drop(['Id'], 1, inplace=True)

    drop_features_with_nan(df)
    drop_by_corr(df)

    # show_heatmap(df)
    show_zoomed_heatmap(df)
    # show_multi_plot(df)






