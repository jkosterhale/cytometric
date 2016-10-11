from collections import defaultdict
import colorlover as cl

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.tools as tools
from plotly.tools import FigureFactory as FF
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

import processing
from settings import RELEVANT_FEATURES, TARGET


def pairwise_plots(data, columns_to_plot, label_column, color_map=None,
                   max_points=False, opacity=.5):
    """Generates all pairwise scatterplots in columns_to_plot

    Points are color-coded according to label_column

    Args:
        data (pd.DataFrame): data to plot
        columns_to_plot (list of str): columns to be paired
        label_column (str): column containing labels for color-coding
        colormap (dict str -> str}: labels mapped to color names
        max_points (int, optional): if specified, plot will contain random subsample
            from data containing this many points

    Returns:
        list of plotly Figure objects containing Scatter plots
    """
    if max_points:
        data_to_plot = data.sample(max_points)
    else:
        data_to_plot = data
    if color_map is None:
        #colors = cl.scales['12']['qual']['Set3'] # 12 colors, like 'rgb(166,206,227)'
        colors = ['rgb(27,158,119)', 'rgb(217,95,2)', 'rgb(117,112,179)']
        labels = data[label_column].unique()
        num_colors = len(labels)
        if num_colors <= len(colors):
            color_map = dict((label, color) for label, color in zip(labels, colors))
        else:
            raise Exception('Too many labels, not enough colors. Make a bigger color_map')
    figures = []
    for row, column1 in enumerate(columns_to_plot):
        for col, column2 in enumerate(columns_to_plot):
            if row < col:
                traces = []
                for label in data_to_plot[label_column].unique():
                    trace = go.Scatter(
                        x = data_to_plot[column1][data_to_plot[label_column]==label],
                        y = data_to_plot[column2][data_to_plot[label_column]==label],
                        name = label,
                        mode = 'markers',
                        marker = dict(size=5,
                                      color=color_map[label],
                                      opacity=opacity))

                    traces.append(trace)

                    layout= go.Layout(hovermode='closest',
                                      xaxis= dict(title=column1,
                                                  ticklen=5,
                                                  zeroline=False,
                                                  gridwidth=2,),
                                      yaxis=dict(title=column2,
                                                 ticklen=5,
                                                 gridwidth=2,),
                                      showlegend= True)
                fig = dict(data=traces, layout=layout)
                figures.append(fig)
    return figures


def histograms(data, columns_to_plot, label_column, show_hist=True):
    """Generates histograms/kde plots for all columns in columns_to_plot

    Args:
        data (pd.DataFrame): data to plot
        columns_to_plot (list of str): columns to generate histograms for
        label_column (str): column containing labels for color-coding

    Returns:
        list of plotly distplots
    """
    labels = sorted(data[label_column].unique())
    figures = []
    for column in columns_to_plot:
        hist_data = []
        for label in labels:
            hist_data.append(data[column][data[label_column]==label])
        fig = FF.create_distplot(hist_data,
                                 labels,
                                 show_hist=show_hist,
                                 show_rug=False)
        figures.append(fig)
    return figures


def generate_pr_curves(data, model, pos_label, features=RELEVANT_FEATURES,
                       target=TARGET, sample_label_column='screen_number'):
    """Creates PR curves for all test sets, using leave-one-screen_number-out CV

    Args:
        data (pd.DataFrame): containing features and target
        model (sklearn classifier, Pipeline, or GridSearchCV):
            Any of:
            - classifier model with fit and predict_proba methods and classes_ attribute
            - Pipeline with such a classifier as its last step
            - GridSearchCV containing such a classifier
        pos_label (str): label of positive outcomes (e.g. 'live')

    Returns:
        list of sklearn.metrics.precision_recall_curve
    """
    if isinstance(model, GridSearchCV):
        model = model.best_estimator_
    pr_curves = {}
    for sample_label in data[sample_label_column].unique():
        # make train/test split, leaving out one sample label
        train = data[sample_label_column] != sample_label
        test = data[sample_label_column] == sample_label
        X_train = data[features][train]
        y_train = data[target][train]
        X_test = data[features][test]
        y_test = data[target][test]
        # fit model on training data
        model.fit(X_train, y_train)
        # get positive label
        if isinstance(model, Pipeline):
            labels = list(model.steps[-1][1].classes_) # assumes classifier is last step of pipeline
        else:
            labels = list(model.classes_)
        pos_label_index = labels.index(pos_label)
        # estimate probabilities on test data
        probas_pred = model.predict_proba(X_test)[:,pos_label_index]
        # compute curve
        pr_curve = precision_recall_curve(y_test, probas_pred, pos_label=pos_label)
        pr_curves['sample label '+str(sample_label)] = pr_curve
    return pr_curves


def plot_pr_curves(pr_curves):
    """Uses plotly to plot precision-recall curves

    Args:
        pr_curves (dict): keys - name of model,
                           values - output from sklearn.metrics.precision_recall curves (precision, recall, thresholds)
    Returns:
        list of figures: one for each model and one with all models overlaid
    """
    # generate figures
    figures = []
    traces = []
    for model, pr in pr_curves.iteritems():
        trace= go.Scatter(
                x= pr[1], # precision
                y= pr[0], # recall
                name = model,
                mode= 'lines',
                text= pr[2]) # threshold
        layout= go.Layout(
            title= 'PR Curve for '+model,
            hovermode= 'closest',
            xaxis= dict(
                title= 'Recall',
                ticklen= 5,
                zeroline= False,
                gridwidth= 2,
            ),
            yaxis=dict(
                title= 'Precision',
                ticklen= 5,
                gridwidth= 2,
                range=[0,1]
            ),
            showlegend= False,)
        traces.append(trace)
        fig= go.Figure(data=[trace], layout=layout)
        figures.append(fig)
    # layout for overlaid PR Curves
    layout= go.Layout(
            title= 'PR Curves',
            hovermode= 'closest',
            xaxis= dict(
                title= 'Recall',
                ticklen= 5,
                zeroline= False,
                gridwidth= 2,
                range=[0,1]
            ),
            yaxis=dict(
                title= 'Precision',
                ticklen= 5,
                gridwidth= 2,
                range=[0,1]
            ),
            showlegend= True,
            legend=dict(x=0,
                        y=0),
            orientation= "h"
        )
    fig = go.Figure(data=traces, layout = layout)

    figures.append(fig)

    return figures
