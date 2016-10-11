"""For training and managing FCS classifier models"""

import cPickle as pkl
import datetime
import os

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, DBSCAN
import fcsparser as fcs

from settings import MODELS_LOCATION, RELEVANT_FEATURES, TARGET
import processing


def fit_and_save_models(dataframe, features, target, models, verbose=True):
    """Fits models and saves them in .pkl files"""
    X = dataframe[features]
    y = dataframe[target]
    while models:
        model = models.pop()
        if verbose: print 'Fitting %s'%str(model)
        model.fit(X, y)
        if verbose: print 'Pickling %s'%str(model)
        model_filename = os.path.join(MODELS_LOCATION, _model_name(model)+'.pkl')
        if verbose: print model_filename
        with open(model_filename, 'wb') as f:
            pkl.dump(model, f)
        del model


def _model_name(model):
    """Infers name of model from GridSearchCV"""
    if isinstance(model, GridSearchCV):
        model = model.estimator
    if isinstance(model, Pipeline):
        model = model.steps[-1][1]
    name =  str(model).split('(')[0] + ' ' + \
            datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S')
    return name

def knn_pipeline(n_neighbors=50):
    """Returns new Pipeline implementing RobustScaler and KNeighborsClassifier"""
    scaler = RobustScaler()
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_pipeline = Pipeline([('scaler', scaler), ('knn', knn)])
    return knn_pipeline


def instantiate_grid_search_models(data, n_jobs=1, scoring='f1_micro',
                                   sample_label_column='screen_number'):
    """Instantiates each GridSearchCV model

    Args:
        data (pd.DataFrame): data to train models on
        n_jobs (int): how many threads to used
        scoring (str): metric to optimize with grid search
        label_column (str): column to use for LeaveOneLabelOut CV

    Returns:
        list of sklearn GridSearchCV models

    """
    cv_settings = {'verbose': 1,
                   'n_jobs': n_jobs,
                   'scoring': scoring,
                   'cv': LeaveOneLabelOut(data[sample_label_column])}

    # K Nearest Neighbors
    knn_param_grid = {'knn__n_neighbors':
                       [25, 50, 75]}
    knn_grid_search_cv = GridSearchCV(knn_pipeline(),
                                      knn_param_grid,
                                      **cv_settings)
    models = [knn_grid_search_cv]
    return models


def cluster_fcs(filename, model, features=RELEVANT_FEATURES,
                return_type='array'):
    '''Trains specified cluster model on FCS file

    KMeans is trained with n_clusters=4.

    DBSCAN is trained with min_samples=100 and eps=5e4.

    Args:
        filename (str): filename of fcs data to be clustered
        model (str): 'kmeans' or 'dbscan'
        return_type (str): 'dataframe' or 'array'

    Returns:
        Either pd.DataFrame with new 'cluster_label' column or an np.array with
        cluster labels
    '''
    # validate input
    if model not in {'kmeans', 'dbscan'}:
        raise Exception('model must be "kmeans" or "dbscan".')
    if return_type not in {'dataframe', 'array'}:
        raise Exception('return_type must be "dataframe" or "array".')

    # train cluster model
    if model.lower()=='kmeans':
        model = KMeans(n_clusters=4)
    elif model.lower()=='dbscan':
        model = DBSCAN(min_samples=100, eps=5e4)
    _, data = fcs.parse(filename)
    model.fit(data[features])
    labels = model.labels_

    if return_type=='dataframe':
        data['cluster_label'] = labels
        return data
    elif return_type=='array':
        return labels


if __name__ == '__main__':
    print 'training models\n'
    dataframe = processing.load_and_process_data()
    models = instantiate_models(dataframe)
    fit_and_save_models(dataframe, RELEVANT_FEATURES, TARGET, models)
    print '\ndone'
