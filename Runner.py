import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(7)


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

    # print(pca.explained_variance_ratio_)
    # print(pca.singular_values_)
    # print(pca.noise_variance_)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(new_data[:, 0], new_data[:, 1], new_data[:, 2], marker='.')
    plt.show()


def show_pca(data):
    pca = PCA(n_components=2)
    pca.fit(data)
    new_data = pca.transform(data)

    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    print(pca.noise_variance_)

    # n_comp = [i for i in range(len(pca.explained_variance_ratio_))]

    # plt.scatter(n_comp, pca.explained_variance_ratio_, marker='.')

    plt.scatter(new_data[:, 0], new_data[:, 1], marker='.')
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(new_data[:, 0], new_data[:, 1], new_data[:, 2], marker='.')
    plt.show()


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


def main():
    df = pd.read_csv(filepath_or_buffer="resources/train.csv").fillna(0)
    encode_labels(df)
    data = df.as_matrix()

    show_pca(data)
    # for kernel in ["linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"]:
    #     show_kernel_pca(data, kernel)
    # show_lda(data)
    # show_mds(data)


if __name__ == "__main__":
    main()