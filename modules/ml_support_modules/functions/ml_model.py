
# Algorithm with the best score from DS notebook
from sklearn.ensemble import AdaBoostClassifier
# Metrics
from sklearn.metrics import accuracy_score, f1_score


def train_model(X_train, y_train):
    """
    Use the train data on AdaBoostClassifier algorithm with default parameters and return it already trained

    Inputs
    ------
    X_train: {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data

    y_train: array-like of shape (n_samples)
        The target values (class labels)    
    Return
    ------
    model
        Trained AdaBoostClassifier: object
    """

    abc_model = AdaBoostClassifier(random_state=42)
    model = abc_model.fit(X_train, y_train)
    return model


def acc_f1_metrics(y, y_pred):
    """
    Validate the trained model using accuracy and f1 score

    Inputs
    ------
    y: 1d array-like, or label indicator array / sparse matrix
        Known labels, binarized

    y_pred: 1d array-like, or label indicator array / sparse matrix
        Predicted labels, binarized

    Returns
    -------
    accurary: float
    f1_score: float
    """
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return accuracy, f1

def predictions(model, X):
    """
    Run model inference and returns the predictions

    Inputs
    ------
    model: the object output from train_model function
        A trained machine learning model
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
        The training inputs samples

    Return
    ------
    predictions: ndarray of shape (n_samples,)
        The predicted classes
    """

    predictions = model.predict(X)
    return predictions
