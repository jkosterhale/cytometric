"""Helper module for implementing probabilistic predictions"""

import pandas as pd
from sklearn.grid_search import GridSearchCV


def predict(data, model, target_filename=None):
    """Wrapper function to get probabilistic predictions

    Returns data frame with new columns containing each label's predicted probability.
    Optionally saves predictions to CSV.

    Args:
        data (pd.DataFrame): dataframe containing observations to predict
        model (sklearn classifier): classifier model used for predictions,
            requires predict_proba method and classes_ attribute.
        target_filename (str, optional): where to save CSV with prediction

    Returns:
        predicted_data, copy of input dataframe with new columns containing each
        label's predicted probability
    """
    if isinstance(model, GridSearchCV):
        model = model.best_estimator_
    predicted_data = data.copy()
    prob_column_names = [n+'_prob' for n in model.classes_]
    predicted_probs = model.predict_proba(data)
    predicted_probs = pd.DataFrame(predicted_probs, columns=prob_column_names)
    predicted_data = pd.concat((predicted_data, predicted_probs), axis=1)
    if target_filename:
        predicted_data.to_csv(target_filename)
    return predicted_data
