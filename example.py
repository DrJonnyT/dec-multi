#Set tensorflow gpu memory to be able to grow
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


from keras_dec.keras_dec import DeepEmbeddingClustering
from mnist.mnist import get_mnist, subsample_mnist

from sklearn.cluster import KMeans

    


X, Y  = get_mnist()

# #Testing dataset with only 100 points
# X100 = X[0:100]
# Y100 = Y[0:100]

# c = DeepEmbeddingClustering(n_clusters=10, input_dim=784,batch_size=256)
# c.initialize(X100, finetune_iters=10000, layerwise_pretrain_iters=5000)
# c.cluster(X100, y=Y100,iter_max=1000)


#Original code with original iterations
c = DeepEmbeddingClustering(n_clusters=10, input_dim=784)
c.initialize(X, finetune_iters=100000, layerwise_pretrain_iters=50000)

#Run KMeans on the initial latent space
kmeans = KMeans(n_clusters=10, n_init=20)
Y_kmeans_0 = kmeans.fit_predict(c.encoder.predict(X))
kmeans_acc_0, _ = c.cluster_acc(Y,Y_kmeans_0)

#Run clustering and interate the latent space
c.cluster(X, y=Y,iter_max=1000)

#Run kmeans again
Y_kmeans_fit = kmeans.fit_predict(c.encoder.predict(X))
kmeans_acc_fit, _ = c.cluster_acc(Y,Y_kmeans_fit)





