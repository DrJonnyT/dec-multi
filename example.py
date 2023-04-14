#Set tensorflow gpu memory to be able to grow
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


from keras_dec.keras_dec import DeepEmbeddingClustering
from mnist.mnist import get_mnist, subsample_mnist
from mnist.plot import plot_mnist_10x10

from sklearn.cluster import KMeans


#Get MNIST data
X, Y  = get_mnist()
    
#%%Run accuracy test with only 100 data points


#Testing dataset with only 100 points
X100, Y100 = subsample_mnist(X,Y,10)


c = DeepEmbeddingClustering(n_clusters=10, input_dim=784)
c.initialize(X100, finetune_iters=100000, layerwise_pretrain_iters=50000)

#Run KMeans on the initial latent space
Y100_kmeans_unfit = KMeans(n_clusters=10, n_init=20).fit_predict(c.encoder.predict(X100))
Y100_kmeans_acc_unfit, _ = c.cluster_acc(Y100,Y100_kmeans_unfit)

c.cluster(X100, y=Y100,iter_max=1000)
#Run kmeans again
Y100_kmeans_fit = KMeans(n_clusters=10, n_init=20).fit_predict(c.encoder.predict(X100))
Y100_kmeans_acc_fit, _ = c.cluster_acc(Y100,Y100_kmeans_fit)



#%%Test plot

fig,ax = plot_mnist_10x10(X, Y, "test title")

#%%


#Original code with original iterations
#This will run on my home 2060 but not on work laptop GPU
#Not yet tried on colab, presumably it would work
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





