import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import load_tabular_data, add_series_features, get_labeled_subset

def main():
    """
    This script does:
    - read tabular data
    - add series features
    - drop unlabeled data points
    - creates a 80-20 train-test split
    - drops columns that are not in test set
    - creates csv files for train and test
    """
    df_train, df_test, _ = load_tabular_data('../data/train.csv', '../data/test.csv', '../data/data_dictionary.csv')

    df_train = add_series_features(df_train, '../data/series_train.parquet')

    df_train = get_labeled_subset(df_train)

    train, test = train_test_split(df_train, test_size=0.2, stratify=df_train["sii"])

    columns_not_in_test = list(set(df_train.columns).difference(set(df_test.columns))) # remove columns not in test
    columns_not_in_test = list(set(columns_not_in_test).difference(set(["enmo_mean", "enmo_std", "light_mean", "light_std", "sii"]))) # keep newly created columns and target
    train = train.drop(labels=columns_not_in_test, axis=1)
    test = test.drop(labels=columns_not_in_test, axis=1)

    train.to_csv("../data/baseline_train.csv", header=True)
    test.to_csv("../data/baseline_test.csv", header=True)


if __name__ == "__main__":
    main()