from keras_dec.functions import linear_assignment, cluster_acc, align_cluster_labels
import numpy as np

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
    
    
    
    
