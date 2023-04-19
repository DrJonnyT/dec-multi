import numpy as np
from sklearn.metrics import adjusted_rand_score
from itertools import combinations

from keras_dec.keras_dec import DeepEmbeddingClustering


def mean_rand_index(df):
    """
    Calculate the mean Rand index for all pairs of columns in a pandas DataFrame `df`,
    where each column represents a different set of cluster labels.
    """
    
    #Remove the actual data labels
    if df.columns[0] == 'labels':
        df.drop('labels',axis=1,inplace=True)
    
    
    # Get all pairs of column names
    column_pairs = list(combinations(df.columns, 2))

    # Calculate the Rand index for each pair of columns
    rand_indices = [adjusted_rand_score(df[pair[0]], df[pair[1]]) for pair in column_pairs]

    # Return the mean Rand index for all pairs of columns
    return np.mean(rand_indices)


def rand_index_arr(df,labels):
    """
    Calculate the Rand index for all columns in a pandas DataFrame 'df'
    (where each column represents a different set of cluster labels), and a
    set of labels 'labels'

    Parameters
    ----------
    df : DataFrame
        Each column is a set of cluster labels. dtype must be int
    labels : array
        cluster labels, of the same length as the number of rows in df.
        dtype must be int

    Returns
    -------
    rand_arr : numpy array
        Array of rand index corresponding to each column in df

    """
    
    #Remove the actual data labels
    if df.columns[0] == 'labels':
        df.drop('labels',axis=1,inplace=True)
    
    rand_arr = [adjusted_rand_score(labels,df[col]) for col in df.columns]
    return rand_arr


def accuracy_arr(df,labels):
    """
    Calculate the accuracy for all columns in a pandas DataFrame 'df'
    (where each column represents a different set of cluster labels), compared 
    to a set of labels 'labels'

    Parameters
    ----------
    df : DataFrame
        Each column is a set of cluster labels. dtype must be int
    labels : array
        cluster labels, of the same length as the number of rows in df.
        dtype must be int

    Returns
    -------
    acc_arr : numpy array
        Array of accuracy corresponding to each column in df

    """
    
    #Remove the actual data labels
    if df.columns[0] == 'labels':
        df.drop('labels',axis=1,inplace=True)
        
    c = DeepEmbeddingClustering(n_clusters=10,input_dim=(784))
    acc_arr = [c.cluster_acc(labels,df[col])[0] for col in df.columns]
    return acc_arr