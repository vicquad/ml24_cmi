import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import os



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