from __future__ import division
from __future__ import print_function

import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "E:\\Machine_Learning"
CHAPTER_ID = "end_to_end_project"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, 'images', CHAPTER_ID)
    if not os.path.exists(path):
        os.makedirs(path)
    print('Saving figure', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(os.path.join(path, fig_id + '.png'), format='png', dpi=300)


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

    house.hist(bins=50, figsize=(20, 15))
    save_fig('attribute_histogram_plots')

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
    house.plot(kind='scatter', x='longitude', y='latitude')
    save_fig("bad_visualization_plot")

    house.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
    save_fig("better_visualization_plot")

    house.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
               s=house['population'] / 100, label='population', figsize=(10, 7),
               c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)

    plt.legend()
    save_fig("housing_prices_scatterplot")

    california_img = mpimg.imread(PROJECT_ROOT_DIR + '\\images\\end_to_end_project\\california.png')
    ax = house.plot(kind="scatter", x="longitude", y="latitude", figsize=(10, 7),
                    s=house['population'] / 100, label="Population",
                    c="median_house_value", cmap=plt.get_cmap("jet"),
                    colorbar=False, alpha=0.4)
    plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
    plt.ylabel("Latitude", fontsize=14)
    plt.xlabel("Longitude", fontsize=14)

    prices = house["median_house_value"]
    tick_values = np.linspace(prices.min(), prices.max(), 11)
    cbar = plt.colorbar()
    cbar.ax.set_yticklabels(["$%dk" % (round(v / 1000)) for v in tick_values], fontsize=14)
    cbar.set_label('Median House Value', fontsize=16)

    plt.legend(fontsize=16)
    save_fig("california_housing_prices_plot")
    plt.show()

    corr_matrix = house.corr()
    # print(corr_matrix)
    attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
    scatter_matrix(house[attributes], figsize=(12, 8))
    save_fig('scatter_matrix_plot')

    house.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)
    plt.axis([0, 16, 0, 550000])
    save_fig('income_vs_house_value_scatterplot')

    # Data clean
    imputer = Imputer(strategy='median')
    house_num = house.drop('ocean_proximity', axis=1)

    # The imputer class has simply computed the median of each attribute
    # and stored the results in its statistics_ instance variable.
    imputer.fit(house_num)
    # print(imputer.statistics_)

    X = imputer.transform(house_num)  # X is a plain numpy array
    house_tr = pd.DataFrame(X, columns=house_num.columns)  # Transform X to a Pandas DataFrame

    # Handing text and categorical sttributes
    encoder = LabelEncoder()
    house_cat = house['ocean_proximity']
    house_cat_encoded = encoder.fit_transform(house_cat)
    # print(house_cat_encoded)

    encoder1 = OneHotEncoder()
    house_cat_1hot = encoder.fit_transform(house_cat_encoded.reshape(-1, 1))
