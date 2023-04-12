from tensorflow.keras.datasets import mnist
import numpy as np

def get_mnist():
    np.random.seed(1234) # set seed for deterministic ordering
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_all = np.concatenate((x_train, x_test), axis = 0)
    Y = np.concatenate((y_train, y_test), axis = 0)
    X = x_all.reshape(-1,x_all.shape[1]*x_all.shape[2])
    
    p = np.random.permutation(X.shape[0])
    X = X[p].astype(np.float32)*0.02
    Y = Y[p]
    return X, Y



def subsample_mnist(X,Y,n,randomize=False):
    """
    Subsample the mnist digits by selecting the first n of each digit

    Parameters
    ----------
    X : array
        mnist digits X (images)
    Y : array
        mnist digits Y (labels)
    n : int
        The number of each digit to output

    Returns
    -------
    Xsub : array
        The first n of mnist digits X
    Ysub : array
        The first n of mnist digits Y

    """
    Xsub = []
    Ysub= []
    
    for label in np.unique(Y):
        #Select the first n points of that particular label and append
        Xsub.append(X[Y==label][0:n])
        Xsub.append(Y[Y==label][0:n])
        
    if randomize:
        p = np.random.permutation(Xsub.shape[0])
        Xsub = Xsub[p]
        Ysub = Ysub[p]
        
    return Xsub, Ysub