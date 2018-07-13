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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

seed = 7
np.random.seed(seed)


def is_residential(ms_zoning):
    return ms_zoning.isin(['RH', 'RL', 'RP', 'RM', 'FV'])


def split_by_zoning(df):
    all_zones = df['MSZoning']

    resident = is_residential(all_zones)
    not_resident = ~resident

    return [df[resident], df[not_resident]]


def show_buy_per_years():
    # test_df = pd.read_csv(filepath_or_buffer="resources/test.csv")
    df = pd.read_csv(filepath_or_buffer="resources/train.csv")

    years_zoning = df[['MSZoning', 'YrSold']]

    group = years_zoning.copy()
    zones = group['MSZoning'].unique()

    by_zones = [group[group['MSZoning'] == z] for z in zones]

    zones_dict = {}
    for z in by_zones:
        index = z['MSZoning'].unique()[0]
        zones_dict[index] = z['YrSold'].as_matrix()
        
    plt.hist(zones_dict.values(), label=zones_dict.keys())
    plt.legend(zones_dict.keys())
    plt.xticks(range(2006, 2011))
    plt.show()


def encode_labels(df):
    column_encoders = {}
    for column_name in df:
        column = df[column_name]
        if column.dtype == 'object' or column.dtype == 'str':
            column = column.astype(str)
            labels = column.unique()

            le = LabelEncoder()
            le.fit(labels)
            column_encoders[column_name] = le

            df[column_name] = column.apply(lambda v: le.transform([v])[0])


def show_kernel_pca(data, kernel):
    pca = KernelPCA(n_components=3, kernel=kernel)
    pca.fit(data)
    new_data = pca.transform(data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(new_data[:, 0], new_data[:, 1], new_data[:, 2], marker='.')
    plt.show()


def show_pca(data):
    pca = PCA(n_components=20)
    pca.fit(data)
    new_data = pca.transform(data)

    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    print(pca.noise_variance_)

    # n_comp = [i for i in range(len(pca.explained_variance_ratio_))]

    # plt.scatter(n_comp, pca.explained_variance_ratio_, marker='.')

    # plt.scatter(new_data[:, 0], new_data[:, 1], marker='.')
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(new_data[:, 0], new_data[:, 1], new_data[:, 2], marker='.')
    # plt.show()

    return new_data


def show_mds(data):
    mds = MDS(n_components=3)
    new_data = mds.fit_transform(data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(new_data[:, 0], new_data[:, 1], new_data[:, 2], marker='.')
    plt.show()


def show_lda(data):
    lda = LatentDirichletAllocation(n_components=3)
    new_data = lda.fit_transform(data)
    # plt.scatter(new_data[:, 0], new_data[:, 1], marker='.')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(new_data[:, 0], new_data[:, 1], new_data[:, 2], marker='.')
    plt.show()


def prepare_data():
    df = pd.read_csv(filepath_or_buffer="resources/train.csv").fillna(0)
    price = df['SalePrice'].as_matrix()

    df = df.drop(['SalePrice', 'Id'], 1)

    encode_labels(df)
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

    accuracy = r2_score(test_price, pred)
    print('accuracy', accuracy)

    # plt.scatter(train_feature, price, color='b', marker='.')
    # plt.scatter(test_feature, test_price, color='r', marker='.')
    # plt.plot(test_feature, pred)
    # plt.show()


def main__():
    df = pd.read_csv(filepath_or_buffer="resources/train.csv").fillna(0)
    price = df['SalePrice'].as_matrix()
    price = minmax_scale(price)
    df = df.drop(['SalePrice', 'Id'], 1)

    encode_labels(df)
    data = df.as_matrix()

    pca_data = show_pca(data)

    data = minmax_scale(pca_data)

    # for kernel in ["linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"]:
    #     show_kernel_pca(data, kernel)
    # show_lda(data)
    # show_mds(data)

    lr = LinearRegression()
    selector = RFE(lr)
    selector.fit(data, price)

    print(selector.support_)
    print(selector.ranking_)

    # for i in range(len(selector.support_)):
    #     if selector.support_[i]:
    #         print(df.iloc[:, [i]].head(1))

    selected_data = selector.transform(data)
    selected_data = selected_data[0:, 0]

    plt.scatter(selected_data, price, marker='.')
    plt.show()


if __name__ == "__main__":
    main()