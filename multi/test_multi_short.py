from multi.comparison import mean_rand_index,rand_index_arr, accuracy_arr, prob_lab_agg
from keras_dec.functions import cluster_acc
from multi.kmeans import kmeans_n_times_csv

import numpy as np
import pandas as pd
import os
from shutil import rmtree

def test_mean_rand_index():
    Xdata = [[0,0],[0.1,0.1],[0.15,0.15],[10,10],[10.1,10.1],[10.2,10.2]]
    Ydata = [58,69,-12,6,np.inf,np.nan]
    csv_path = "./temp/kmeans2.csv"
    df_kmeans = kmeans_n_times_csv(Xdata, 10, 2, csv_path, newcsv=True, labels=Ydata)
    
    mean_rand = mean_rand_index(df_kmeans)
    #The mean rand should be 1 because you should always come up with the same
    #set of labels for this Xdata
    assert mean_rand == 1
    
    
def test_rand_index_arr():
    df1 = pd.DataFrame()
    df1['kmeans_1'] = [0,0,1,1,2,2]
    df1['fgadfg'] = [2,2,1,1,0,0]
    df1['kmeans3'] = [0,1,1,2,2,0]
    
    labels = np.array([0,0,1,1,2,2])
    
    rand_arr = rand_index_arr(df1,labels)
    assert np.array_equal(rand_arr,[1,1,-0.25])
    
    
def test_accuracy_arr():
    df1 = pd.DataFrame()
    df1['kmeans_1'] = [0,0,1,1,2,2]
    df1['fgadfg'] = [2,2,1,1,0,0]
    df1['kmeans3'] = [0,1,1,2,2,0]
    
    labels = np.array([0,0,1,1,2,2])
    
    acc_arr = accuracy_arr(df1,labels)
    assert np.array_equal(acc_arr,[1,1,0.5])
    
    
def test_prob_lab_agg():
    df_labels = pd.DataFrame()
    df_labels['labels_1'] = [0,0,0,1,1,1]
    df_labels['labels_2'] = [1,1,1,0,0,0]
    labels_pla = prob_lab_agg(df_labels)
    assert cluster_acc(labels_pla,df_labels['labels_1'])[0] == 1
    
    #Add in some more labels
    df_labels['labels_3'] = [0,0,0,1,1,1]
    df_labels['labels_4'] = [1,0,0,1,0,0]
    labels_pla = prob_lab_agg(df_labels)
    assert cluster_acc(labels_pla,df_labels['labels_1'])[0] == 1
    
#After all tests completed, delete the temp directory
def pytest_sessionfinish(session, exitstatus):
    if os.path.isdir("./temp"):
        rmtree("./temp")
        
def test_tidy():
    try:
        rmtree("./temp")
    except:
        raise Exception('Failed to delete temp folder in test_multi')
    
    