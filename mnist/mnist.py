from tensorflow.keras.datasets import mnist
import numpy as np
from random import shuffle

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



def subsample_mnist(X,Y,n10,randomize=False):
    """
    Subsample the mnist digits by selecting the first n of each digit

    Parameters
    ----------
    X : array
        mnist digits X (images)
    Y : array
        mnist digits Y (labels)
    n10 : int
        The number of each digit to output
    randomize: bool
        If true, it will randomise Xsub and Ysub AFTER it selects the digits

    Returns
    -------
    Xsub : array
        The first n of mnist digits X
    Ysub : array
        The first n of mnist digits Y

    """
    Xsub = []
    Ysub= []
    #p is the point number
    p = np.arange(0,X.shape[0])
    psub=[]
        
    for label in np.unique(Y):
        #Identify the first n points of this particular label
        psub.append(p[Y==label][0:n10])
            
    psub = np.sort(np.ravel(psub))
    
    if randomize:
        shuffle(psub)
    
    #Select data subset
    Xsub = X[psub]
    Ysub = Y[psub]
                
        
    return Xsub, Ysub


