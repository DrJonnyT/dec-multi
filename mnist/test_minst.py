from mnist.mnist import get_mnist, subsample_mnist, subsample_digits
from mnist.plot import plot_mnist_10x10
import numpy as np

def test_get_mnist():
    X, Y  = get_mnist()
    assert np.shape(X) == (70000, 784)
    assert np.shape(Y) == (70000,)  
    
    
def test_subsample_mnist():
    X,Y = get_mnist()
    n = 10
    X100, Y100 = subsample_mnist(X,Y,n)
    
    assert np.shape(X100) == (10*n,784)
    assert np.shape(Y100) == (10*n,)
    
    X100r, Y100r = subsample_mnist(X,Y,n,randomize=True)
    assert np.shape(X100r) == (10*n,784)
    assert np.shape(Y100r) == (10*n,)
    
    #Check randomized and non-randomized data are the same if you sort them
    assert np.array_equal( np.sort(X100r[Y100r == 0],axis=0), np.sort(X100[Y100 == 0],axis=0) )
    
    
def test_plot_mnist_10x10():
    X,Y = get_mnist()
    
    fig,ax = plot_mnist_10x10(X, Y, "test title")
    #Just see if it gets this far without an error
    
    
    
def test_subsample_digits():
    X,Y = get_mnist()
    
    #First do 100 digits, balanced
    Xsub, Ysub = subsample_digits(X,Y,balanced=True)
    assert len(Xsub) == 100
    assert len(Ysub) == 100
    
    #Now 10000 unbalanced
    Xsub, Ysub = subsample_digits(X,Y,n_digits=10000)
    assert len(Xsub) == 10000
    assert len(Ysub) == 10000
    #Check you dont have balanced numbers of each digit
    assert np.array_equal( np.histogram(Ysub) , np.tile(10,10)) is False
    
    #Now all the digits
    Xsub, Ysub = subsample_digits(X,Y,n_digits=0)
    assert len(Ysub) == 70000
    