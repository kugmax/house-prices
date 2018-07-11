import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def is_residential(ms_zoning):
    return ms_zoning.isin(['RH', 'RL', 'RP', 'RM', 'FV'])


def split_by_zoning(df):
    all_zones = df['MSZoning']

    resident = is_residential(all_zones)
    not_resident = ~resident

    return [df[resident], df[not_resident]]


def main():
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


if __name__ == "__main__":
    main()