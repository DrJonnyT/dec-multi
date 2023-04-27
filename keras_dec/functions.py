from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd

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



def align_cluster_labels(labels1,labels2):
    """
    Take two sets of cluster labels and assign them so the nearest equivalents
    use the same labels.    

    Parameters
    ----------
    labels1 : array
        Array of integer labels.
    labels2 : array
        Array of integer labels.

    Returns
    -------
    labels2_aligned : numpy array
        Version of labels2 aligned to have a similar numbering system as labels1.

    """
    
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)
    labels2_aligned = np.zeros(labels2.shape)
    
    
    #Check y_true and y_pred are the same size
    assert labels1.size == labels2.size, "labels1 and labels2 are not the same size"
    
    #Generate the confusion matrix
    D = max(labels1.max(), labels2.max())+1
    cm = np.zeros((D, D), dtype=np.int64)
    for i in range(labels2.size):
        cm[labels2[i], labels1[i]] += 1
        
    #Work out which labels in labels2 correspond to which labels in labels1
    ind = linear_assignment(cm.max() - cm)
    
    #Make a dictionary to map the input labels2 to the aligned versions
    my_dict = {}
    for row in ind:
        my_dict[row[0]] = row[1]
    
    #Map labels2 to the aligned version
    labels2_aligned = np.vectorize(my_dict.get)(labels2)
      
    return labels2_aligned


def modal_labels(df_labels):
    df_modal_labels = df_labels.mode(axis=1)
    #At this stage we have a dataframe with modal labels for most rows in the
    #first column. The rest of the columns are nans except when a row has an
    #unambiguous mode when two or more clusters have the same number of points.
    #Then it would list all the equally most frequent labels in the columns.
    #So we need to randomly sample these to get the fairest set of labels.
    
    ds_modal_labels = pd.Series(index=df_labels.index,dtype='float')
    
    for row in df_modal_labels.index:
        row_nonans = df_modal_labels.loc[row].dropna()
        
        if len(row_nonans) == 1:
            ds_modal_labels.loc[row] = row_nonans.values
        else:
            #Randomly sample a column
            ds_modal_labels.loc[row] = row_nonans.sample().iloc[0]
            
    return ds_modal_labels.values.astype('int')