import pandas as pd
from sklearn.model_selection import train_test_split
from utils import load_tabular_data

def main():
    """
    This script does:
    - read tabular data
    - creates a 80-20 train-test split
    - drops columns that are not in test set
    - creates csv files for train and test
    """

    df_train, df_test, _ = load_tabular_data("train.csv", "test.csv", "data_dictionary.csv")

    unlabeled_data = df_train[df_train['sii'].isna()]  # Keep rows where 'sii' is NaN
    labeled_data = df_train[df_train['sii'].notna()]  # Keep rows where 'sii' is not NaN

    train_labeled, test_labeled = train_test_split(labeled_data, test_size=0.2, stratify=labeled_data["sii"])
    train_unlabeled, test_unlabeled = train_test_split(unlabeled_data, test_size=0.2)

    train = pd.concat([train_labeled, train_unlabeled], axis=0).sample(frac=1).reset_index(drop=True) # shuffle
    test = pd.concat([test_labeled, test_unlabeled], axis=0).sample(frac=1).reset_index(drop=True) # shuffle

    columns_not_in_test = list(set(df_train.columns).difference(set(df_test.columns))) # remove columns not in test
    columns_not_in_test = list(set(columns_not_in_test).difference(set(["sii"]))) # keep target
    train = train.drop(labels=columns_not_in_test, axis=1)
    test = test.drop(labels=columns_not_in_test, axis=1)
    
    print("# train samples: ", len(train), ", # test samples: ", len(test))
    train.to_csv("tabdata_train.csv", header=True)
    test.to_csv("tabdata_test.csv", header=True)

if __name__ == "__main__":
    main()
