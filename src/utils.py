import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import cohen_kappa_score
import os
import tsfel



def load_tabular_data(train_path, test_path, dict_path):
    """
    Load the training, test, and dictionary data from the given paths.
    
    Parameters
    ----------
    train_path : str
        Path to the training data.
    test_path : str
        Path to the test data.
    dict_path : str
        Path to the dictionary data.
    
    Returns
    -------
    X_train : DataFrame
        Training data.
    y_train : Series
        Target labels.
    X_test : DataFrame
        Test data.
    data_dict : DataFrame
        Dictionary data.
    """
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)
    data_dict = pd.read_csv(dict_path)
    
    return X_train, X_test, data_dict


def add_series_features(X, series_path):
    """
    Add features from the series data to the input data.

    Parameters
    ----------
    X : DataFrame
        Input data.
    series_path : str
        Path to the series folder.

    Returns
    -------
    X : DataFrame
        Input data with added features.
    """
    X[['enmo_mean', 'enmo_std', 'light_mean', 'light_std']] = np.nan

    for id in X['id'].values:
        # first check if the file exists
        if not os.path.exists(f'{series_path}\id={id}'):
            continue
        
        # read the file and extract the features
        df_series = pd.read_parquet(f'{series_path}\id={id}', engine='pyarrow')
        X.loc[X['id'] == id, 'enmo_mean'] = df_series['enmo'].mean()
        X.loc[X['id'] == id, 'enmo_std'] = df_series['enmo'].std()
        X.loc[X['id'] == id, 'light_mean'] = df_series['light'].mean()
        X.loc[X['id'] == id, 'light_std'] = df_series['light'].std()
    
    return X

def extract_series_features(X, series_path):
    """
    First clean the series data and then extract features from it.

    Parameters
    ----------
    X : DataFrame
        Input data.
    series_path : str
        Path to the series folder.

    Returns
    -------
    X : DataFrame
        Input data with added features.
    """
    cfg = tsfel.get_features_by_domain()

    for id in X['id'].values:
        # first check if the file exists
        if not os.path.exists(f'{series_path}\id={id}'):
            continue

        df_series = pd.read_parquet(f'{series_path}\id={id}', engine='pyarrow')
        df_series = df_series.dropna()
        df_series = df_series[df_series['non_wear_flag'] != 1]


    return None
    

def evaluate_model(model, X, y):
    """
    Evaluate the given model using quadratic weighted kappa and mean accuracy.

    Parameters
    ----------
    model : object
        Model to be evaluated.
    X : array-like of shape (n_samples, n_features)
        Input data.

    Returns
    -------

    kappa : float
        Quadratic weighted kappa.
    mean_accuracy : float
        Mean accuracy.
    """
    y_pred = model.predict(X)
    kappa = cohen_kappa_score(y, y_pred, weights='quadratic')
    mean_accuracy = model.score(X, y)
    
    return kappa, mean_accuracy


def get_labeled_subset(data):
    """
    Get all data points with labels from a given train set.

    Parameters
    ----------
    X : DataFrame
        Input data, including target column "sii".

    Returns
    -------
    X : DataFrame
        Input data, including only rows where target is not missing.

    """
    data = data.dropna(subset=["sii"])
    return data 


def impute_tabdata(data, test, dict_path):
    """
    Imputes missing values in the given set of tabular data.

    Parameters
    ----------
    data : DataFrame
        Input data, including target column "sii". Can be train or test set.
    
    dict_path: str
        Path to data dictionary csv file.

    Returns
    -------
    X : DataFrame
        Imputed input data.

    """
    data_dict = pd.read_csv(dict_path)

    # Separate target value
    target = data["sii"]
    data = data.drop(columns="sii")

    # Identify categorical integer fields
    categorical_int_features = data_dict[data_dict['Type'].str.contains('categorical int', na=False)]['Field'].tolist()
    
    # Drop PCIAT features as they are not part of train data
    categorical_int_features = [s for s in categorical_int_features if "PCIAT" not in s]

    # Extract numerical features
    numerical_features = data_dict[
        data_dict['Type'].str.contains('float|int', na=False) & 
        ~data_dict['Type'].str.contains('categorical|str', na=False)
    ]['Field'].tolist()

    # Drop PCIAT features as they are not part of train data
    numerical_features = [s for s in numerical_features if "PCIAT" not in s]
    numerical_features.extend(['enmo_mean', 'enmo_std', 'light_mean', 'light_std'])

    # Extract categorical features
    categorical_features = data_dict[
        data_dict["Type"].str.contains("categorical|str", na=False)
    ]["Field"].tolist()
    
    # Drop PCIAT features as they are not part of train data
    categorical_features = [s for s in categorical_features if "PCIAT" not in s]

    # ensure categorical features are of type str
    data[categorical_features] = data[categorical_features].astype(str)

    # impute numerical features with KNNImputer
    # num_imputer = KNNImputer(n_neighbors=3)
    num_imputer = SimpleImputer(strategy="mean")
    data[numerical_features] = num_imputer.fit_transform(data[numerical_features])

    # impute numerical features with SimpleImputer - most frequent value
    cat_imputer = SimpleImputer(strategy="most_frequent")
    imputed_categorical = cat_imputer.fit_transform(data[categorical_features])
    
    # Convert the result back to a DataFrame with the original column names
    data[categorical_features] = pd.DataFrame(imputed_categorical, columns=categorical_features, index=data.index)
    
    # Combine data and target again
    data["sii"] = target

    # Impute test data
    test[numerical_features] = num_imputer.transform(test[numerical_features])
    imputed_categorical_test = cat_imputer.transform(test[categorical_features])
    test[categorical_features] = pd.DataFrame(imputed_categorical_test, columns=categorical_features, index=test.index)

    return data, test

def oversample_tabdata(data):
    """
    Oversamples minority classes for given tabular data.
 
    Parameters
    ----------
    data : DataFrame
        Input data, including target column "sii". 
        This should be the label-propagated train set with no missing labels.

    Returns
    -------
    X_resampled : DataFrame
        Oversampled input data.

    """

    y = data["sii"]
    X = data.drop(columns="sii")

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_resampled["sii"] = y_resampled

    return X_resampled


def label_propagation(data, model):
    """
    Propagates labels for the given tabular data using the given model.

    Parameters
    ----------
    data : DataFrame
        Input data, including target column "sii". 
        This should be the train set with missing labels.

    model : object
        Model to be used for label propagation.

    Returns
    -------
    data : DataFrame
        Input data with propagated labels.

    """

    # Separate data with missing labels
    data_missing = data[data["sii"].isna()]
    data_labeled = data.dropna(subset=["sii"])

    # Separate target value
    target = data_labeled["sii"]
    data_labeled = data_labeled.drop(columns="sii")

    # Fit the model
    model.fit(data_labeled, target)

    # Predict missing labels
    data_missing.loc[:, "sii"] = model.predict(data_missing.drop(columns="sii"))

    # Combine data with propagated labels
    data_labeled["sii"] = target
    data = pd.concat([data_missing, data_labeled])

    return data