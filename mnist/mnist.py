from tensorflow.keras.datasets import mnist
import numpy as np
from random import shuffle

def get_mnist():
    """
    Download a copy of the mnist digits data from tensorflow.keras

    Returns
    -------
    X : array
        Array of images.
    Y : Array
        Array of labels.

    """
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
        mnist digits X (images).
    Y : array
        mnist digits Y (labels).
    n10 : int
        The number of each digit to output.
    randomize: bool
        If true, it will randomise Xsub and Ysub AFTER it selects the digits.

    Returns
    -------
    Xsub : array
        The first n of mnist digits X.
    Ysub : array
        The first n of mnist digits Y.

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



def subsample_digits(X,Y,n_digits=100,balanced=False):
    """
    Subsample MNIST digits. A more advanced version of subsample_mnist, capable
    of balanced (equal numbers of each digit) and unbalanced sampling.

    Parameters
    ----------
    X : array
        Data to be subsampled (probably MNIST images).
    Y : array
        Labels to be subsampled (probably MNIST labels).
    resample : TYPE, optional
        DESCRIPTION. The default is False.
    n_digits : int, optional
        The number of digits to sample. If set to <=0, it will run with the
        full MNIST dataset and ignore the resample flag. The default is 100.
    balanced : TYPE, optional
        Take a balanced sample of n_digits/10 copies of each digit, rather than
        a fully random sample. The default is False.
    n10 : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    Xsub : array
        Subsampled version of X
    Ysub : array
        Subsampled version of Y

    """
    
    #The number of each digit to sample if using a balanced dataset
    if balanced is True:
        n10 = int(n_digits/10)
    
    
    #Select the digits
    if n_digits<=0 or n_digits==70000:
        #Do not subsample data
        Xsub = X
        Ysub = Y
    else:
        #Subsample data
        #Empty lists for the subsampled data
        Xsub = np.zeros((0, 784))
        Ysub = np.zeros(0,dtype='int')
        
        if balanced is True:
            # Select 10 instances of each digit (0-9) at random
            for digit in range(10):
                indices = np.where(Y == digit)[0]
                indices = np.random.choice(indices, size=n10, replace=False)
                Xsub = np.vstack((Xsub,X[indices]))
                Ysub = np.append(Ysub,Y[indices])
        else:
            #Select n_digits at random
            indices = np.random.randint(0,len(X),n_digits)
            Xsub = X[indices]
            Ysub = Y[indices]
            
    return Xsub, Ysub