import numpy as np
from sklearn.metrics import adjusted_rand_score
from itertools import combinations


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