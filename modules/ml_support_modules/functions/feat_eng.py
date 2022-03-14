import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def read_data(path):
    """
    Use pandas module to read a csv file.
    Inputs
    ------
    path: Any valid string path is acceptable. The string could be a URL. Valid URL schemes include:
    http, ftp, s3, gs, and file. For file URLs, a host is expected.
    Return
    ------
    A comma-separated values (csv) file is returned as two-dimensional data structure with labeled axes
    """
    df = pd.read_csv(path)
    return df


def process_data(
    X,label=None, training=True, encoder=None):

    """
    Process the data before use it on a Machine Learning model.

    This can be used for training and for testing data (inference)

    Inputs
    ------
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
        Data containing the features and targets, it will be split into X and y during the process
    label: str
        The name of the column in X that contain the values the model should predict (y)
        If None, it will return y as an empty array
    training: boolean
        Indicates if you functions is on training mode or inference mode
    encoder: sklearn.preprocessing StandardScaler
        Only use it if training=False to not have leaking from test data
    
    Return
    ------
    X: ndaray array of shape (N_samples, n_featuress_new)
        Processed data standardized
    y: a numpy array
        Array with the values from the label column if label is not None, otherwise returns an empty array
    encoder: sklearn.preprocessing StandardScaler
        Trained StandardScaler encoder is training is True, otherwise returns the encoder passed

    """

    # Drop Id column if already exists
    X.drop(["Id"], axis= 1, inplace= True, errors= "ignore")

    if label is not None:

        X["binary_y"] = X[label] > 5
        X.drop([label], inplace=True, axis=1)
        y = X["binary_y"]

        X = X.drop(["binary_y"], axis=1)
    else:
        y = np.array([])

    if training is True:
        encoder = StandardScaler()
        X = encoder.fit_transform(X)
    else:
        X = encoder.transform(X)

    return X, y, encoder
    