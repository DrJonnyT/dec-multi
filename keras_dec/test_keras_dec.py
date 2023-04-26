from keras_dec.functions import linear_assignment, cluster_acc
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
    