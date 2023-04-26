from scipy.optimize import linear_sum_assignment
import numpy as np

def linear_assignment(cost_matrix):
    """
    linear_assignment function to give clusters meaningful labels.
        
    In this context, this function takes clusters with arbitrary labels and 
    compares them to the test labels of the samples in that cluster. It then
    relabels the clusters with the label that best matches the cluster.
    For example, you might have a cluster that is mostly 7's, but because
    k-means labels are arbitrary, that cluster was labelled 'cluster 3.
    This function will look at the data within 'cluster 3' and decide that it
    should be called 'cluster 7' instead, because it is mostly 7's.
    
    The original from sklearn is now deprecated, so using scipy v instead, so really
    all this function does is reformat the output to emulate the scipy version.

    Parameters
    ----------
    cost_matrix : numpy.ndarray
        Square cost matrix of cluster labels

    Returns
    -------
    numpy.ndarray
        The cost matrix but rearranged to different cluster labels.

    """
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def cluster_acc(y_true, y_pred):
    """
    Cluster accuracy function from DeepEmbeddedClustering class

    Parameters
    ----------
    y_true : numpy array
        Array of true cluster labels.
    y_pred : numpy array
        Array of predicted cluster labels.

    Returns
    -------
    accuracy : float
        Cluster label accuracy.
    w : numpy array
        Cost matrix.

    """
    #Check y_true and y_pred are the same size
    assert y_pred.size == y_true.size, "y_true and y_pred are not the same size"
    #Generate the cost matrix
    D = max(y_pred.max(), y_true.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    accuracy = sum([w[i, j] for i, j in ind])*1.0/y_pred.size
    return accuracy, w