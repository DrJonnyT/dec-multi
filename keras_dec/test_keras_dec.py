from keras_dec.functions import linear_assignment, cluster_acc, align_cluster_labels
from keras_dec.keras_dec import DeepEmbeddingClustering
from mnist.mnist import get_mnist, subsample_mnist

import numpy as np
import tensorflow as tf
import os
import random

def test_DeepEmbeddingClustering():
    #Download and subsample mnist dataset
    X,Y = get_mnist()
    
    #First 10 digits of the mnist dataset
    X,Y = subsample_mnist(X,Y,10,randomize=False)
    
    SEED = 1337
    
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+
    
    
    
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)   
    
    tf.config.experimental.enable_op_determinism()
    
    #Fix random seed so it gives the same result every time
    g = tf.Graph()
    accuracy = []
    np.random.seed(1337)
    y_pred = []
    with g.as_default():
        tf.random.set_seed(1337)
        
        #Run clustering
        c = DeepEmbeddingClustering(n_clusters=10,
                                    input_dim=np.shape(X)[1])
        c.initialize(X, finetune_iters=1000,
                     layerwise_pretrain_iters=500,
                     verbose=0)
        y_pred = c.cluster(X,Y, iter_max=100,save_interval=0)
        accuracy = c.accuracy[-1]


    assert accuracy == 0.58
    assert cluster_acc(Y,y_pred)[0] == 0.58
    
    labels = np.array([4, 4, 1, 5, 9, 5, 8, 9, 7, 0, 2, 8, 1, 1, 1, 9, 9, 4, 1, 7, 5, 5,
           9, 8, 5, 1, 0, 0, 0, 6, 3, 3, 6, 0, 1, 1, 4, 8, 5, 2, 3, 2, 9, 1,
           4, 7, 6, 5, 2, 8, 9, 3, 0, 1, 3, 2, 3, 1, 4, 1, 1, 7, 8, 8, 1, 9,
           1, 4, 2, 1, 0, 9, 0, 7, 1, 4, 8, 0, 4, 8, 7, 4, 2, 0, 4, 1, 4, 8,
           8, 7, 8, 4, 7, 2, 2, 7, 8, 6, 3, 2], dtype=int)
    assert np.array_equal(labels,y_pred)
    


def test_linear_assignment():
         
    cost_mtx_in = np.array([[0,3],[3,0]])
    cost_mtx_out = linear_assignment(cost_mtx_in)
    cost_mtx_out_check = np.array([[0,0],[1,1]])
    
    assert np.array_equal(cost_mtx_out,cost_mtx_out_check)
    
    
def test_cluster_acc():
    labels1 = np.array([0,0,1,1,1])
    labels2 = np.array([1,1,0,0,0])
    labels3 = np.array([0,0,1,1,0])
    
    assert cluster_acc(labels1,labels1)[0] == 1
    assert cluster_acc(labels1,labels2)[0] == 1
    assert cluster_acc(labels1,labels3)[0] == 0.8
    

def test_align_cluster_labels():
    #Try first with some standard labels
    labels1 = np.array([0,1,2,3,4])
    labels2 = np.array([4,1,2,3,7])
    
    assert np.array_equal(align_cluster_labels(labels1,labels2),labels1)
    
    #Try with random integers
    rand_integers1 = np.random.randint(low=0, high=10, size=10000)
    rand_integers2 = np.random.randint(low=0, high=10, size=10000)
    
    #Just in case they are exactly the same...
    if np.array_equal(rand_integers1,rand_integers2):
        rand_integers2 = np.random.randint(low=0, high=10, size=10000)
    
    rand_integers2_aligned = align_cluster_labels(rand_integers1,rand_integers2)
    
    #Make sure that aligning them doesn't change the number of each label
    hist2,_ = np.histogram(rand_integers2)
    hist2_aligned,_ = np.histogram(rand_integers2_aligned)
    
    assert np.array_equal(np.sort(hist2),np.sort(hist2_aligned))
    
    
    
    
