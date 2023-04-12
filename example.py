from keras_dec.keras_dec import DeepEmbeddingClustering
from mnist.mnist import get_mnist, subsample_mnist



    


X, Y  = get_mnist()

#Testing dataset with only 100 points
X100 = X[0:10000]
Y100 = Y[0:10000]

c = DeepEmbeddingClustering(n_clusters=10, input_dim=784,batch_size=256)
c.initialize(X100, finetune_iters=10000, layerwise_pretrain_iters=5000)
c.cluster(X100, y=Y100,iter_max=1000)


#Original code with original iterations
# c = DeepEmbeddingClustering(n_clusters=10, input_dim=784)
# c.initialize(X, finetune_iters=100000, layerwise_pretrain_iters=50000)
# c.cluster(X, y=Y)