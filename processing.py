"""Tools to import FCS files"""

import glob
import os
from fnmatch import fnmatch
import cPickle as pkl
from collections import namedtuple, defaultdict
from copy import deepcopy

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

import fcsparser as fcs
from settings import DATA_LOCATION, DROPBOX_LOCATION, MODELS_LOCATION, \
                     RELEVANT_FEATURES, LABELS, TARGET, \
                     EVENT_IDENTIFYING_COLUMNS

# FCS files should have this filename format
FILENAME_TEMPLATE = 'screen_*.fcs'
DROPBOX_FILENAME_TEMPLATE = 'Well_*.fcs'
DROPBOX_DIRNAME_ATTRIBUTES = ['well', 'cell_plate', 'screen']

# FILENAME_ATTRIBUTES is used as a basis for iteration
FILENAME_ATTRIBUTES = {'screen_number', 'cell_plate_number', 'well_number',
                       'cell_type'}

# FCSFilenameAttributes is used to parse filename directly upon import
FCSFilenameAttributes = namedtuple('fcs_filename', ['screen',
                                                    'screen_number',
                                                    'cell',
                                                    'plate',
                                                    'cell_plate_number',
                                                    'well',
                                                    'well_number',
                                                    'cell_type'])

def get_sample_dirnames(directory):
    '''Returns all .fcs dirnames (e.g. "/folder/filename.fcs") in directory/subdirectories'''
    found_dirs = os.walk(directory)
    dirnames = []
    for dir_ in found_dirs:
        directory, _, filenames = dir_
        for filename_with_extension in filenames:
            filename, extension = os.path.splitext(filename_with_extension)
            if extension=='.fcs':
                dirname = os.path.join(directory, filename_with_extension)
                dirnames.append(dirname)
    return dirnames


def _get_dirname_attributes(dirname):
    dirname, ext = os.path.splitext(dirname)
    elements = dirname.split('/')
    attributes = {}
    for attr in DROPBOX_DIRNAME_ATTRIBUTES:
        for element in elements[::-1]: # iterates backwards because relevant elements are near the end
            if attr + '_' in element.lower(): # add underscore to match dirname
                attr = attr.strip('_')
                attributes[attr] = element
                break
    return attributes


def load_fcs_files(location=DATA_LOCATION, filename_template=FILENAME_TEMPLATE,
                   verbose=True):
    """Loads fcs files as pandas dataframes

    Attributes from filename are added as columns to the dataframes.

    Args:
        location (str): folder containing fcs files
        filename_template (str): format of filenames, using * for wildcard

    Returns:
        list of pandas DataFrames containing fcs data
    """
    filenames = glob.glob(os.path.join(location, filename_template))
    dataframes = []
    for filename in filenames:

        # load fcs file from disk into DataFrame
        meta_data, dataframe = fcs.parse(filename)
        dataframe._filename = filename
        # add columns for screen #, plate #, well #, and cell type
        try:
            filename_with_ext = os.path.basename(filename)
            if verbose: print 'Loading: %s'%filename_with_ext
            filename2 = os.path.splitext(filename_with_ext)[0]
            filename_attributes = FCSFilenameAttributes(*filename2.split('_'))
        except: 
            try: 
                import re
                x = re.split('/|_|.fcs',filename)
                filename_attributes = FCSFilenameAttributes(*[x[8],x[9],x[10],x[11],x[12],x[19],x[20],"unlabeled"])
            except: 
                filename_attributes = FCSFilenameAttributes(*[x[8],x[9],x[10],x[11],x[12],x[15],x[16],x[18]])
        for attr in FILENAME_ATTRIBUTES:
            value = filename_attributes.__getattribute__(attr).lower()
            if attr == 'cell_type':
                if value in LABELS:
                    dataframe['cell_type'] = value
                    dataframe['is_live'] = value == 'live'
                    dataframe['is_dead'] = value == 'dead'
                    dataframe['is_debris'] = value == 'debris'
                    dataframe['is_blast'] = value == 'blast'
                    dataframe['is_healthy'] = value == 'healthy'
                    dataframe.cell_type_is_labeled = True
                else:
                    dataframe[attr] = 'unlabeled'
                    dataframe.cell_type_is_labeled = False
            else:
                dataframe[attr] = value
        dataframe['filename'] = filename
        dataframes.append(dataframe)
    return dataframes


def outlier_data(dataframe, features_to_scale, overwrite=False,threshold=3.5):
    """    If overwrite is False, for each feature 'X' in features_to_scale,
    scale_features adds 'X_scaled' column. Otherwise column 'X' is overwritten.

    Args:
        dataframe (pd.DataFrame) - dataframe containing data to be scaled
        features_to_scale (list of str): features that should be scaled
        overwrite (bool): whether to replace the original data with scaled data
    """
    dataframe_cut = dataframe
    totalsize = dataframe.shape[0]
    for feature in features_to_scale:
        points = dataframe_cut[feature]
        median = np.median(points)
        diff = (points - median)**2
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)
        modified_z_score = 0.6745 * diff / med_abs_deviation  # 75% of the data 
        x= modified_z_score < threshold 
        
        dataframe_cut = dataframe_cut.loc[x]

    cut = float(dataframe_cut[feature].shape[0])/float(totalsize)
    return dataframe_cut, cut 
                    

            

def scale_data(dataframe, features_to_scale, overwrite=False):
    """Applies sklearn.preprocessing.RobustScaler

    If overwrite is False, for each feature 'X' in features_to_scale,
    scale_features adds 'X_scaled' column. Otherwise column 'X' is overwritten.

    Args:
        dataframe (pd.DataFrame) - dataframe containing data to be scaled
        features_to_scale (list of str): features that should be scaled
        overwrite (bool): whether to replace the original data with scaled data
    """
    for feature in features_to_scale:
        #scaler = RobustScaler()
        scaler = RobustScaler()
        scaled_feature = scaler.fit_transform(dataframe[feature].as_matrix().reshape(-1, 1))
        if overwrite:
            dataframe[feature]=scaled_feature
        else:
            dataframe[feature+'_scaled']=scaled_feature

            
def boxcox_data(dataframe, features_to_scale, overwrite=False):
    """Applies sklearn.preprocessing.RobustScaler

    If overwrite is False, for each feature 'X' in features_to_scale,
    scale_features adds 'X_scaled' column. Otherwise column 'X' is overwritten.

    Args:
        dataframe (pd.DataFrame) - dataframe containing data to be scaled
        features_to_scale (list of str): features that should be scaled
        overwrite (bool): whether to replace the original data with scaled data
    """
    from scipy.stats import boxcox
    for feature in features_to_scale:
        dataframe.loc[dataframe[feature] <= 0,feature] = 1 
        scaled_feature, lamb = boxcox(dataframe[feature])
        if overwrite:
            dataframe[feature]=scaled_feature
        else:
            dataframe[feature+'_bc_'+str(round(lamb,2))]=scaled_feature

def log_data(dataframe, features_to_scale, overwrite=False):
    """Applies sklearn.preprocessing.RobustScaler

    If overwrite is False, for each feature 'X' in features_to_scale,
    scale_features adds 'X_scaled' column. Otherwise column 'X' is overwritten.

    Args:
        dataframe (pd.DataFrame) - dataframe containing data to be scaled
        features_to_scale (list of str): features that should be scaled
        overwrite (bool): whether to replace the original data with scaled data
    """
    for feature in features_to_scale:
        dataframe.loc[dataframe[feature] <= 0,feature] = 1 
        scaled_feature = np.log(dataframe[feature])
        if overwrite:
            dataframe[feature]=scaled_feature
        else:
            dataframe[feature+'_log']=scaled_feature
            
            
            

def concatenate_dataframes_inclusive(dataframes):
    """Concatenates dataframes from FCS files where cell_type_is_labeled==True

     retains ALL columns shared in all FCS files

    Args:
        dataframes (list of pd.DataFrame): dataframes to be concatenated

    Returns:
        DataFrame containing all rows from input dataframes
    """
    labeled_dataframes = filter(lambda x: x.cell_type_is_labeled, dataframes)
    concatenated_dataframe = pd.concat(labeled_dataframes, join='outer',
                                       ignore_index=True)
    return concatenated_dataframe
            
def concatenate_dataframes(dataframes):
    """Concatenates dataframes from FCS files where cell_type_is_labeled==True

    Only retains columns shared in all FCS files

    Args:
        dataframes (list of pd.DataFrame): dataframes to be concatenated

    Returns:
        DataFrame containing all rows from input dataframes
    """
    labeled_dataframes = filter(lambda x: x.cell_type_is_labeled, dataframes)
    concatenated_dataframe = pd.concat(labeled_dataframes, join='inner',
                                       ignore_index=True)
    return concatenated_dataframe


def add_dapi_columns(df, df_with_dapi):
    """Helper function to add DAPI columns to a dataframe

    Performs left join on settings.EVENT_IDENTIFYING_COLUMNS. Makes a copy of df.

    Args:
        df (pd.DataFrame): dataframe to get new columns
        df_with_dapi (pd.DataFrame): dataframe with 'DAPI A' and 'DAPI H' columns

    Returns:
        copy of df with 'DAPI A' and 'DAPI H' columns
    """
    columns_to_keep = set(EVENT_IDENTIFYING_COLUMNS + ['DAPI A', 'DAPI H'])
    orig_columns = set(deepcopy(df_with_dapi.columns))
    for column in orig_columns:
        if column not in columns_to_keep:
            df_with_dapi.drop(column, axis=1, inplace=True, errors='ignore')

    merged_df = pd.merge(df, df_with_dapi, how='left', on=EVENT_IDENTIFYING_COLUMNS,
                         copy=True)
    return merged_df

def add_labeled_columns(df, df_with_dapi):
    """Helper function to add DAPI columns to a dataframe

    Performs left join on settings.EVENT_IDENTIFYING_COLUMNS. Makes a copy of df.

    Args:
        df (pd.DataFrame): dataframe to get new columns
        df_with_dapi (pd.DataFrame): dataframe with 'DAPI A' and 'DAPI H' columns

    Returns:
        copy of df with 'DAPI A' and 'DAPI H' columns
    """

    drops = ["Time", "Width"]
    keeps = list(df_with_dapi.columns[~df_with_dapi.columns.isin(drops)])
    columns_to_keep = set(EVENT_IDENTIFYING_COLUMNS + keeps)
    print columns_to_keep
    shared_cols = pd.merge(pd.DataFrame(df.columns.values), pd.DataFrame(df_with_dapi.columns.values), how = "inner")
    shared_cols =    list(shared_cols[0])
    shared_cols.remove("filename") 
    shared_cols.remove("cell_type")   
    orig_columns = set(deepcopy(df_with_dapi.columns))
    for column in orig_columns:
        if column not in columns_to_keep:
            df_with_dapi.drop(column, axis=1, inplace=True, errors='ignore')

    merged_df = pd.merge(df, df_with_dapi, how='outer', on=shared_cols,
                         copy=False)
    return merged_df


def load_and_process_data(path=DATA_LOCATION,template = FILENAME_TEMPLATE,
                          features_to_scale=RELEVANT_FEATURES,
                          overwrite=True,
                          verbose=True):
    """Loads fcs files from disk and returns single dataframe

    Args:
        path (str): location of fcs files
        features_to_scale (list of str): features to apply scale_data to
        overwrite (bool): whether to overwrite columns with scaled columns
        verbose (bool): whether to print names of fcs files being loaded

    Returns:
        dataframe containing all data from fcs files
    """
    # load data from disk
    dataframes = load_fcs_files(filename_template = template, verbose=False)
    # combine the dataframes, keeping shared columns
    #dataframe = concatenate_dataframes(dataframes)
    dataframe = dataframes
    return dataframe


def _infer_model_name(model):
    if isinstance(model, GridSearchCV):
        model = model.estimator
    if isinstance(model, Pipeline):
        model = model.steps[-1][1]
    name = str(model).split('(')[0]
    return name


def load_models(name='', path=MODELS_LOCATION, verbose=True):
    """Loads all .pkl files in path or model with 'name'"""
    if name:
        filepathnames = [name]
    elif path:
        template = os.path.join(path, '*.pkl')
        filepathnames = glob.glob(template)
    else:
        raise Exception('No models found')
    models = []
    for name in filepathnames:
        if verbose: print 'loading %s'%name
        with open(name, 'rb') as f:
            models.append(pkl.load(f))
    return models


def select_best_models(models, scoring=None):
    """Returns model of each type with highest score

    Args:
        models (list of sklearn GridSearchCV models)
        scoring (optional): target metric. If scoring is specified, only models
                            using this metric are returned

    Returns:
        best models (dict name -> model), best_score (dict name -> score)
    """
    if scoring:
        models = filter(lambda x:x.scoring==scoring, models)
    models = sorted(models, key=lambda x:x.best_score_, reverse=True)
    best_models, best_scores = {}, {}
    for model in models:
        model_name = _infer_model_name(model)
        if model_name not in best_models:
            best_models[model_name] = model
            best_scores[model_name] = model.best_score_

    return best_models, best_scores
