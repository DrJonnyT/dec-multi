from mnist.mnist import get_mnist, subsample_mnist
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
    