from keras_dec.functions import linear_assignment, cluster_acc, align_cluster_labels
from keras_dec.keras_dec import DeepEmbeddingClustering
from mnist.mnist import get_mnist, subsample_mnist

import numpy as np
import tensorflow as tf
import os
import random
from pytest import MonkeyPatch

def test_DeepEmbeddingClustering(monkeypatch):
    
    #Make some fake test images
    #Black images
    X0 = (np.zeros([3,784]) + np.random.normal(0,0.05,[3,784])).clip(min=0)
    #Grey images
    X50 = (np.ones([3,784]) * np.random.normal(0.5,0.05,[3,784])).clip(min=0,max=1)
    
    X = np.concatenate([X0,X50],axis=0)
    Y = np.array([0,0,0,1,1,1])
    
    #Need to set random seeds and other settings so it is the same every time
    #you run the test
    #This is a useful resource:
    #https://github.com/NVIDIA/framework-reproducibility/blob/master/doc/d9m/tensorflow_status.md
    
    #MonkeyPatch is a way of temporarily setting os.environ variables
    with MonkeyPatch.context() as mp:
    
        #Required random fixing
        SEED = 1234
        mp.setenv('TF_DETERMINISTIC_OPS','1')
        
        #Equivalent:
        #os.environ['TF_DETERMINISTIC_OPS'] = '1'
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        
        #Not required(?)
        #os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+
        #tf.config.experimental.enable_op_determinism()    
        
        g = tf.Graph()    
        with g.as_default():
            tf.random.set_seed(SEED)
            
            #Run clustering
            c = DeepEmbeddingClustering(n_clusters=2,
                                        input_dim=np.shape(X)[1])
            c.initialize(X, finetune_iters=1000,
                          layerwise_pretrain_iters=500,
                          verbose=0)
            y_pred = c.cluster(X,Y, iter_max=100,save_interval=0)
            accuracy = c.accuracy[-1]
        
        
    #Test that output is as expected
    assert accuracy == 1
    assert cluster_acc(Y,y_pred)[0] == 1



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
    
    
    
    
