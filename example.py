from keras_dec.keras_dec import DeepEmbeddingClustering
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



#def subsample_mnist(n10):
    
    

X, Y  = get_mnist()

#Testing dataset with only 100 points
X100 = X[0:10000]
Y100 = Y[0:10000]

c = DeepEmbeddingClustering(n_clusters=10, input_dim=784,batch_size=50)
c.initialize(X100, finetune_iters=1000, layerwise_pretrain_iters=500)
c.cluster(X100, y=Y100,iter_max=1000)


#Original code with original iterations
# c = DeepEmbeddingClustering(n_clusters=10, input_dim=784)
# c.initialize(X, finetune_iters=100000, layerwise_pretrain_iters=50000)
# c.cluster(X, y=Y)