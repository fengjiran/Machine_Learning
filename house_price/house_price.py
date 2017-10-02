from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def load_housing_data(path='E:\\Machine_Learning\\house_price'):
    csv_path = os.path.join(path, 'housing.csv')
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


if __name__ == '__main__':
    house = load_housing_data()
    print(len(house))
    print(house.head())
    print(house.info())
    print(house['ocean_proximity'].value_counts())
    print(house.describe())

    # plt.hist(house['median_income'], bins=50)
    # plt.show()

    house['income_cat'] = np.ceil(house['median_income'] / 1.5)
    house['income_cat'].where(house['income_cat'] < 5, 5.0, inplace=True)
    print(house['income_cat'].value_counts())

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(house, house['income_cat']):
        strat_train_set = house.loc[train_index]
        strat_test_set = house.loc[test_index]

    print(house['income_cat'].value_counts() / len(house))

    for dataset in (strat_train_set, strat_test_set):
        dataset.drop(['income_cat'], axis=1, inplace=True)

    house = strat_train_set.copy()

    house.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
               s=house['population'] / 100, label='population',
               c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)

    # house.hist(bins=50, figsize=(20, 15))
    plt.show()

    corr_matrix = house.corr()
    print(corr_matrix)

    # train_set, test_set = split_train_test(house, 0.2)
    # print(len(train_set), len(test_set))
