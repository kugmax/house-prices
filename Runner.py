from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from nltk.classify import svm
from scipy.stats import stats
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import AdaBoostRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, Ridge, ElasticNet
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC, OneClassSVM
from sklearn.tree import DecisionTreeRegressor

from Tools import encode_labels, show_sale_price_statistic

target_column = 'SalePrice'
hight_coreletion_with_target = [target_column, 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
max_nan_percentage = 0.15
seed = 7
np.random.seed(seed)

"""
Steps:
 + transform skewed data by log(1 + x)

 + fit mean or most frequent by logic
 + transform categorical number to categorical
 - transform categorical by LabelEncoder
 + transform categorical by dummies
 use ensemble methods 
 + Out liars 
 + if feature has > 15% missing data delete it
"""


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

    k = 10
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

    cm = np.corrcoef(df[cols].values.T)

    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                     xticklabels=cols.values)
    plt.show()


def show_multi_plot(df):
    sns.set()
    sns.pairplot(df[hight_coreletion_with_target], size=2)
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

    for column_name in df:
        if column_name == target_column: continue
        nan_percentage[column_name] = get_percentage_missing(df[column_name])

    column_to_drop = list(map(lambda x: x[0],
                              filter(lambda x: x[1] > max_nan_percentage, nan_percentage.items()))
                          )

    df.drop(labels=column_to_drop, axis=1, inplace=True)


def transform_number_categorical_to_str_categorical(df):
    number_categorical = ['MSSubClass', 'OverallQual', 'OverallCond']

    for column in number_categorical:
        is_null = list(
                        filter(lambda x: x,
                               df[column].isnull()
                               )
                    )
        if is_null:
            raise Exception('There is a null')

    for column_name in number_categorical:
        df[column_name] = df[column_name].astype(str)


def fill_nan(df):
    for column_name in df:
        if column_name == target_column: continue

        column = df[column_name]
        if column.isnull().sum() > 0:
            if column.dtype == 'object' or column.dtype == 'str':
                df[column_name].fillna(column.mode()[0], inplace=True)
            else:
                df[column_name].fillna(column.mean(), inplace=True)


def drop_by_corr(df):
    culumns = ['GarageArea', 'GarageYrBlt', 'GarageType', 'GarageCond', 'GarageFinish', 'GarageQual',

               'MoSold', 'MiscVal', 'LotFrontage', 'Utilities', 'Condition2', 'FireplaceQu',
               'Exterior2nd', 'SaleType',
               
               'TotRmsAbvGrd', '1stFlrSF']
    df.drop(labels=culumns, axis=1, inplace=True)


def log_transform(df):
    for col_name in hight_coreletion_with_target:
        try:
            if df[col_name].dtype not in ['object', 'str']:
                df[col_name] = np.log1p(df[col_name])
        except Exception:
            pass


def rmsle_cv(model, x_train, y_train):
    n_folds = 5
    kf = KFold(n_folds, shuffle=True, random_state=seed).get_n_splits(x_train)
    rmse = np.sqrt(-cross_val_score(model, x_train, y_train, scoring="neg_mean_squared_error", cv=kf))
    return rmse


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def rmse(y_true, y_pred):
    diff = y_pred - y_true
    sum_sq = sum(diff ** 2)
    n = len(y_pred)

    return np.sqrt(sum_sq / n)


def predict(X_train, Y_trin, X_test):
    lasso_model = Lasso(alpha=0.0005, random_state=seed)
    lasso_model.fit(X_train, Y_trin)

    y_test = lasso_model.predict(X_test)

    y_test = np.exp(y_test)

    index = [i for i in range(1461, 2920)]
    y_test = pd.DataFrame(y_test, index=index)
    y_test.columns = ['SalePrice']

    y_test.to_csv('submission.csv', index_label='Id')
    # y_test.plot.hist(bins=20)


def check_score(X, Y):
    lasso_model = Lasso(alpha=0.0005, random_state=seed)
    score = rmsle_cv(lasso_model, X, Y)
    print("Lasso score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

    # linear_model = LinearRegression()
    # score = rmsle_cv(linear_model, X, Y)
    # print("LinearRegression score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    #
    # ridge_model = Ridge(alpha=0.0005, random_state=seed)
    # score = rmsle_cv(ridge_model, X, Y)
    # print("Ridge score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    #
    # elastic_net_model = ElasticNet(alpha=0.0005, random_state=seed)
    # score = rmsle_cv(elastic_net_model, X, Y)
    # print("ElasticNet score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    #
    # ada_boost_model = AdaBoostRegressor()
    # score = rmsle_cv(ada_boost_model, X, Y)
    # print("AdaBoostRegressor score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    #
    # decision_tree_model = DecisionTreeRegressor(min_samples_split=20)
    # score = rmsle_cv(decision_tree_model, X, Y)
    # print("DecisionTreeRegressor score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    #
    # k_neighborn_model = KNeighborsRegressor()
    # score = rmsle_cv(k_neighborn_model, X, Y)
    # print("KNeighborsRegressor score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


def show_outliers(outliers, y, y_pred, z):

    plt.figure(figsize=(15, 5))
    ax_131 = plt.subplot(1, 3, 1)
    plt.plot(y, y_pred, '.')
    plt.plot(y.loc[outliers], y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('y')
    plt.ylabel('y_pred')

    ax_132 = plt.subplot(1, 3, 2)
    plt.plot(y, y - y_pred, '.')
    plt.plot(y.loc[outliers], y.loc[outliers] - y_pred.loc[outliers], 'ro')
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('y')
    plt.ylabel('y - y_pred')

    ax_133 = plt.subplot(1, 3, 3)
    z.plot.hist(bins=50, ax=ax_133)
    z.loc[outliers].plot.hist(color='r', bins=50, ax=ax_133)
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('z')
    plt.show()


def find_outliers(model, X, y, sigma=3):
    model.fit(X, y)
    y_pred = pd.Series(model.predict(X), index=y.index)

    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    z = (resid - mean_resid) / std_resid
    outliers = z[abs(z) > sigma].index

    return outliers, y, y_pred, z


def drop_outlines(train):
    for column_name in hight_coreletion_with_target:
        column = train[column_name]
        X = np.array(column, dtype=float).reshape(-1, 1)
        outliers, y, y_pred, z = find_outliers(Lasso(alpha=0.0005, random_state=seed), X, train['SalePrice'])

        train.drop(outliers, inplace=True)


if __name__ == "__main__":
    start_drop = ['Id']

    df_train = pd.read_csv(filepath_or_buffer="resources/train.csv")
    df_test = pd.read_csv(filepath_or_buffer="resources/test.csv")

    id_train = len(df_train.index)

    df_all = pd.concat([df_train, df_test], sort=False)
    df_all.drop(start_drop, 1, inplace=True)

    drop_by_corr(df_all)
    drop_features_with_nan(df_all)
    transform_number_categorical_to_str_categorical(df_all)
    fill_nan(df_all)
    encode_labels(df_all)
    log_transform(df_all)

    x_train = df_all.iloc[:id_train, :]
    x_test = df_all.iloc[id_train:, :]
    drop_outlines(x_train)

    # show_sale_price_statistic(x_train)
    # show_multi_plot(x_train)
    # show_heatmap(x_train)
    # show_zoomed_heatmap(x_train)

    id_train = len(x_train)
    df_all = pd.concat([x_train, x_test], sort=False)

    y_train = df_all['SalePrice']
    df_all.drop(['SalePrice'], 1, inplace=True)
    x_train = pd.get_dummies(df_all)
    x_train = StandardScaler().fit_transform(x_train)

    x_test = x_train[id_train:, :]
    x_train = x_train[:id_train, :]
    y_train = y_train[:id_train]

    check_score(x_train, y_train)

    predict(x_train, y_train, x_test)






