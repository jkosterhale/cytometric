from sklearn.cluster import KMeans
import numpy as np

def mixed_kmeans(data, morphology_features, expression_features):
    """Clusters FCS events using mixed k-means model

    Model fits two k-means models with k=3, one using morphology features and
    another using expression features. Final cluster labels are created by
    combining the cluster labels from these two models.

    Function assumes that any transformations (such as log, logicle, etc.) have
    already been applied

    Args:
        data (pd.DataFrame): FCS data
        morphology_features (list of str): features for first model
        expression_features (list of str): features for second model

    Returns:
        ndarray of cluster labels
    """
    # fit morphology model
    kmeans_morphology = KMeans(n_clusters=3)
    morphology_labels = kmeans_morphology.fit_predict(data[morphology_features])
    # fit expression model
    kmeans_expression= KMeans(n_clusters=3)
    expression_labels = kmeans_expression.fit_predict(data[expression_features])
    # create combined cluster labels by string-concatenating both label sets
    labels = np.core.defchararray.add(morphology_labels.astype(str),
                                      expression_labels.astype(str))
    return labels
