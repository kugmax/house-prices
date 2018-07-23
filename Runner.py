from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from nltk.classify import svm
from scipy.stats import stats
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, Ridge, ElasticNet
from sklearn.model_selection import KFold, train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC, OneClassSVM
from sklearn.tree import DecisionTreeRegressor

from Tools import encode_labels, show_sale_price_statistic

target_column = 'SalePrice'
# hight_coreletion_with_target = [target_column, 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
#                                 'FullBath', 'YearBuilt']

hight_coreletion_with_target = [target_column, 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
                                'FullBath', 'YearBuilt']

# hight_coreletion_with_target = [target_column, 'OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearRemodAdd',
#                                 'Fireplaces', 'Foundation', 'MasVnrArea']
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
    columns = ['SalePrice', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2',
               'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']

    corrmat = df[columns].corr()

    k = len(columns) #10
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
    corr = df.corr()

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
    df["n_MSSubClass"] = df.MSSubClass.map({'180': 1,
                                            '30': 2, '45': 2,
                                            '190': 3, '50': 3, '90': 3,
                                            '85': 4, '40': 4, '160': 4,
                                            '70': 5, '20': 5, '75': 5, '80': 5, '150': 5,
                                            '120': 6, '60': 6})

    df["n_MSZoning"] = df.MSZoning.map({'C (all)': 1, 'RH': 2, 'RM': 2, 'RL': 3, 'FV': 4})

    df["n_Neighborhood"] = df.Neighborhood.map({'MeadowV': 1,
                                                'IDOTRR': 2, 'BrDale': 2,
                                                'OldTown': 3, 'Edwards': 3, 'BrkSide': 3,
                                                'Sawyer': 4, 'Blueste': 4, 'SWISU': 4, 'NAmes': 4,
                                                'NPkVill': 5, 'Mitchel': 5,
                                                'SawyerW': 6, 'Gilbert': 6, 'NWAmes': 6,
                                                'Blmngtn': 7, 'CollgCr': 7, 'ClearCr': 7, 'Crawfor': 7,
                                                'Veenker': 8, 'Somerst': 8, 'Timber': 8,
                                                'StoneBr': 9,
                                                'NoRidge': 10, 'NridgHt': 10})

    df["n_Condition1"] = df.Condition1.map({'Artery': 1,
                                            'Feedr': 2, 'RRAe': 2,
                                            'Norm': 3, 'RRAn': 3,
                                            'PosN': 4, 'RRNe': 4,
                                            'PosA': 5, 'RRNn': 5})

    df["n_BldgType"] = df.BldgType.map({'2fmCon': 1, 'Duplex': 1, 'Twnhs': 1, '1Fam': 2, 'TwnhsE': 2})

    df["n_HouseStyle"] = df.HouseStyle.map({'1.5Unf': 1,
                                               '1.5Fin': 2, '2.5Unf': 2, 'SFoyer': 2,
                                               '1Story': 3, 'SLvl': 3,
                                               '2Story': 4, '2.5Fin': 4})

    df["n_Exterior1st"] = df.Exterior1st.map({'BrkComm': 1,
                                                 'AsphShn': 2, 'CBlock': 2, 'AsbShng': 2,
                                                 'WdShing': 3, 'Wd Sdng': 3, 'MetalSd': 3, 'Stucco': 3, 'HdBoard': 3,
                                                 'BrkFace': 4, 'Plywood': 4,
                                                 'VinylSd': 5,
                                                 'CemntBd': 6,
                                                 'Stone': 7, 'ImStucc': 7})

    df["n_MasVnrType"] = df.MasVnrType.map({'BrkCmn': 1, 'None': 1, 'BrkFace': 2, 'Stone': 3})

    df["n_ExterQual"] = df.ExterQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    df["n_Foundation"] = df.Foundation.map({'Slab': 1,
                                               'BrkTil': 2, 'CBlock': 2, 'Stone': 2,
                                               'Wood': 3, 'PConc': 4})

    df["n_BsmtQual"] = df.BsmtQual.map({'Fa': 2, 'None': 1, 'TA': 3, 'Gd': 4, 'Ex': 5})

    df["n_BsmtExposure"] = df.BsmtExposure.map({'None': 1, 'No': 2, 'Av': 3, 'Mn': 3, 'Gd': 4})

    df["n_Heating"] = df.Heating.map({'Floor': 1, 'Grav': 1, 'Wall': 2, 'OthW': 3, 'GasW': 4, 'GasA': 5})

    df["n_HeatingQC"] = df.HeatingQC.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    df["n_KitchenQual"] = df.KitchenQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    df["n_Functional"] = df.Functional.map(
        {'Maj2': 1, 'Maj1': 2, 'Min1': 2, 'Min2': 2, 'Mod': 2, 'Sev': 2, 'Typ': 3})

    df["n_FireplaceQu"] = df.FireplaceQu.map({'None': 1, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    df["n_GarageType"] = df.GarageType.map({'CarPort': 1, 'None': 1,
                                               'Detchd': 2,
                                               '2Types': 3, 'Basment': 3,
                                               'Attchd': 4, 'BuiltIn': 5})

    df["oGarageFinish"] = df.GarageFinish.map({'None': 1, 'Unf': 2, 'RFn': 3, 'Fin': 4})

    df["oPavedDrive"] = df.PavedDrive.map({'N': 1, 'P': 2, 'Y': 3})

    df["oSaleType"] = df.SaleType.map({'COD': 1, 'ConLD': 1, 'ConLI': 1, 'ConLw': 1, 'Oth': 1, 'WD': 1,
                                           'CWD': 2, 'Con': 3, 'New': 3})

    df["oSaleCondition"] = df.SaleCondition.map(
        {'AdjLand': 1, 'Abnorml': 2, 'Alloca': 2, 'Family': 2, 'Normal': 3, 'Partial': 4})


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
    culumns = ['MSSubClass', 'MSZoning', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle',
               'Exterior1st', 'MasVnrType', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtExposure',
               'Heating', 'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
               'PavedDrive', 'SaleType', 'SaleCondition']

    # df.drop(labels=culumns, axis=1, inplace=True)
    # 0.0985(0.0045)
    # 0.0976(0.0040)
    pass


def make_new_feature(df):
    df["TotalHouse"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalArea"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"] + df["GarageArea"]

    df["+_TotalHouse_OverallQual"] = df["TotalHouse"] * df["OverallQual"]
    df["+_GrLivArea_OverallQual"] = df["GrLivArea"] * df["OverallQual"]
    df["+_n_MSZoning_TotalHouse"] = df["n_MSZoning"] * df["TotalHouse"]
    df["+_n_MSZoning_OverallQual"] = df["n_MSZoning"] + df["OverallQual"]
    df["+_n_MSZoning_YearBuilt"] = df["n_MSZoning"] + df["YearBuilt"]
    df["+_n_Neighborhood_TotalHouse"] = df["n_Neighborhood"] * df["TotalHouse"]
    df["+_n_Neighborhood_OverallQual"] = df["n_Neighborhood"] + df["OverallQual"]
    df["+_n_Neighborhood_YearBuilt"] = df["n_Neighborhood"] + df["YearBuilt"]
    df["+_n_BsmtFinSF1_OverallQual"] = df["BsmtFinSF1"] * df["OverallQual"]

    df["-_n_Functional_TotalHouse"] = df["n_Functional"] * df["TotalHouse"]
    df["-_n_Functional_OverallQual"] = df["n_Functional"] + df["OverallQual"]
    df["-_LotArea_OverallQual"] = df["LotArea"] * df["OverallQual"]
    df["-_TotalHouse_LotArea"] = df["TotalHouse"] + df["LotArea"]
    df["-_n_Condition1_TotalHouse"] = df["n_Condition1"] * df["TotalHouse"]
    df["-_n_Condition1_OverallQual"] = df["n_Condition1"] + df["OverallQual"]

    df["Bsmt"] = df["BsmtFinSF1"] + df["BsmtFinSF2"] + df["BsmtUnfSF"]
    df["Rooms"] = df["FullBath"] + df["TotRmsAbvGrd"]
    df["PorchArea"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
    df["TotalPlace"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"] + \
                       df["GarageArea"] + df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]


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
    # lasso_model = Lasso(alpha=0.0005, random_state=seed)
    lasso_model = Lasso(alpha=0.001, random_state=seed)
    lasso_model.fit(X_train, Y_trin)

    y_test = lasso_model.predict(X_test)

    y_test = np.exp(y_test)

    index = [i for i in range(1461, 2920)]
    y_test = pd.DataFrame(y_test, index=index)
    y_test.columns = ['SalePrice']

    y_test.to_csv('submission.csv', index_label='Id')
    # y_test.plot.hist(bins=20)

# TODO: need try make new feature by + something
# TODO: GarageYrBlt - the max value is 2207, this is obviously wrong since the data is only until 2010.
# TODO: try screw from scipy.special import boxcox1p, boxcox_normmax
# TODO: prevent overfiting its about outlines Leave-One-Out methodology with OLS import statsmodels.api as sm
# TODO: prevent overfiting remove 97% 1 or 0 after doing pd.get_dummies.
# TODO from xgboost import XGBRegressor may improve
# TODO: check and delete
# print(features['Street'].value_counts())
# print('-----')
# print(features['Utilities'].value_counts())
# print('-----')
# print(features['CentralAir'].value_counts())
# print('-----')
# print(features['PavedDrive'].value_counts())


def check_score(X, Y, x_columns):
    # model = Lasso(random_state=seed)
    #
    # param_grid = {'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]}
    #
    # # hyper_param_optimize = RandomizedSearchCV(model, n_jobs=4, cv=5, param_distributions=param_grid)
    # hyper_param_optimize = GridSearchCV(model, n_jobs=4, cv=5, param_grid=param_grid)
    # hyper_param_optimize.fit(X, Y)
    # params = hyper_param_optimize.get_params()
    # print(params)
    # print(hyper_param_optimize.best_estimator_)

    lasso_model = Lasso(alpha=0.001, random_state=seed)
    score = rmsle_cv(lasso_model, X, Y)
    print("Lasso score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

    lasso_model.fit(X, Y)
    lasso_coef = pd.DataFrame({"Lasso coef":lasso_model.coef_}, index=x_columns)
    lasso_coef.sort_values("Lasso coef", ascending=False)
    print(lasso_coef.sort_values("Lasso coef", ascending=False))

    lasso_coef[lasso_coef["Lasso coef"] != 0].sort_values("Lasso coef").plot(kind="barh", figsize=(15, 25))
    plt.xticks(rotation=90)
    plt.show()


    # linear_model = LinearRegression()
    # score = rmsle_cv(linear_model, X, Y)
    # print("LinearRegression score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

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

    # gbr_model = GradientBoostingRegressor(learning_rate=0.001, n_estimators=10000, random_state=seed)
    # score = rmsle_cv(gbr_model, X, Y)
    # print("GradientBoostingRegressor score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    #
    # decision_tree_model = DecisionTreeRegressor(min_samples_split=20)
    # score = rmsle_cv(decision_tree_model, X, Y)
    # print("DecisionTreeRegressor score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    #
    # k_neighborn_model = KNeighborsRegressor()
    # score = rmsle_cv(k_neighborn_model, X, Y)
    # print("KNeighborsRegressor score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

def show_lasso_coef(model):
    # model.coef
    pass


def show_outliers(outliers, y, y_pred, z, column_name):
    if len(outliers) <= 0:
        print('No outlines for ', column_name)
        return

    plt.figure(figsize=(15, 5))
    ax_131 = plt.subplot(1, 3, 1)
    plt.plot(y, y_pred, '.')
    plt.plot(y.loc[outliers], y_pred.loc[outliers], 'ro')
    plt.title(column_name)
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('y')
    plt.ylabel('y_pred')

    ax_132 = plt.subplot(1, 3, 2)
    plt.plot(y, y - y_pred, '.')
    plt.plot(y.loc[outliers], y.loc[outliers] - y_pred.loc[outliers], 'ro')
    plt.title(column_name)
    plt.legend(['Accepted', 'Outlier'])
    plt.xlabel('y')
    plt.ylabel('y - y_pred')

    ax_133 = plt.subplot(1, 3, 3)
    z.plot.hist(bins=50, ax=ax_133)
    z.loc[outliers].plot.hist(color='r', bins=50, ax=ax_133)
    plt.title(column_name)
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

        # show_outliers(outliers, y, y_pred, z, column_name)

        train.drop(outliers, inplace=True)


if __name__ == "__main__":
    start_drop = ['Id']

    df_train = pd.read_csv(filepath_or_buffer="resources/train.csv")
    df_test = pd.read_csv(filepath_or_buffer="resources/test.csv")

    id_train = len(df_train.index)

    df_all = pd.concat([df_train, df_test], sort=False)
    df_all.drop(start_drop, 1, inplace=True)

    transform_number_categorical_to_str_categorical(df_all)
    make_new_feature(df_all)
    drop_by_corr(df_all)
    drop_features_with_nan(df_all)
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
    x_columns = x_train.columns

    x_train = StandardScaler().fit_transform(x_train)

    x_test = x_train[id_train:, :]
    x_train = x_train[:id_train, :]
    y_train = y_train[:id_train]

    check_score(x_train, y_train, x_columns)

    predict(x_train, y_train, x_test)






