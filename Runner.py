from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, Ridge, ElasticNet
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor

from Tools import encode_labels, show_sale_price_statistic

hight_coreletion_with_target = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
max_nan_percentage = 0.15
seed = 7
np.random.seed(seed)

"""
Steps:
 understend witch feature more effective by LassoCV

 + transform skewed data by log(1 + x)

 + fit mean or most frequent by logic
 + transform categorical number to categorical
 - transform categorical by LabelEncoder
 + transform categorical by dummies
 use ensemble methods 
 Out liars 
 + if feature has > 15% missing data delete it
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

    for column_name in df:
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
        column = df[column_name]
        if column.isnull().sum() > 0:
            if column.dtype == 'object' or column.dtype == 'str':
                df[column_name].fillna(column.mode()[0], inplace=True)
            else:
                df[column_name].fillna(column.mean(), inplace=True)


def drop_by_corr(df):
    culumns = ['GarageArea', 'TotRmsAbvGrd', '1stFlrSF']
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


def predict(X, Y):
    # TODO: need remove outlines

    X = StandardScaler().fit_transform(X)

    linear_model = LinearRegression()
    score = rmsle_cv(linear_model, X, Y)
    print("LinearRegression score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

    lasso_model = Lasso(alpha=0.0005, random_state=seed)
    score = rmsle_cv(lasso_model, X, Y)
    print("Lasso score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

    ridge_model = Ridge(alpha=0.0005, random_state=seed)
    score = rmsle_cv(ridge_model, X, Y)
    print("Ridge score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

    elastic_net_model = ElasticNet(alpha=0.0005, random_state=seed)
    score = rmsle_cv(elastic_net_model, X, Y)
    print("ElasticNet score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

    ada_boost_model = AdaBoostRegressor()
    score = rmsle_cv(ada_boost_model, X, Y)
    print("AdaBoostRegressor score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

    decision_tree_model = DecisionTreeRegressor(min_samples_split=20)
    score = rmsle_cv(decision_tree_model, X, Y)
    print("DecisionTreeRegressor score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

    k_neighborn_model = KNeighborsRegressor()
    score = rmsle_cv(k_neighborn_model, X, Y)
    print("KNeighborsRegressor score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

    # stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR),
    #                                                  meta_model=lasso)
    #
    # score = rmsle_cv(stacked_averaged_models)
    # print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    #
    # stacked_averaged_models.fit(train.values, y_train)
    # stacked_train_pred = stacked_averaged_models.predict(train.values)
    # stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
    # print(rmsle(y_train, stacked_train_pred))
    #
    # model_xgb.fit(train, y_train)
    # xgb_train_pred = model_xgb.predict(train)
    # xgb_pred = np.expm1(model_xgb.predict(test))
    # print(rmsle(y_train, xgb_train_pred))
    #
    # model_lgb.fit(train, y_train)
    # lgb_train_pred = model_lgb.predict(train)
    # lgb_pred = np.expm1(model_lgb.predict(test.values))
    # print(rmsle(y_train, lgb_train_pred))
    #
    # print('RMSLE score on train data:')
    # print(rmsle(y_train, stacked_train_pred * 0.70 +
    #             xgb_train_pred * 0.15 + lgb_train_pred * 0.15))
    #
    # ensemble = stacked_pred * 0.70 + xgb_pred * 0.15 + lgb_pred * 0.15


if __name__ == "__main__":
    # start_drop = ['Id', 'SaleCondition', 'SaleType']
    start_drop = ['Id']

    df = pd.read_csv(filepath_or_buffer="resources/train.csv")
    df.drop(start_drop, 1, inplace=True)

    drop_features_with_nan(df)
    drop_by_corr(df)
    transform_number_categorical_to_str_categorical(df)
    fill_nan(df)
    encode_labels(df)
    log_transform(df)

    # show_sale_price_statistic(df)
    # show_multi_plot(df)
    # show_heatmap(df)
    # show_zoomed_heatmap(df)

    df_test = pd.read_csv(filepath_or_buffer="resources/test.csv")
    df_test.drop(start_drop, 1, inplace=True)

    drop_features_with_nan(df_test)
    drop_by_corr(df_test)
    transform_number_categorical_to_str_categorical(df_test)
    fill_nan(df_test)
    encode_labels(df_test)
    log_transform(df_test)

    y_train = df['SalePrice']
    df.drop(['SalePrice'], 1, inplace=True)
    x_train = pd.get_dummies(df)

    predict(x_train, y_train)






