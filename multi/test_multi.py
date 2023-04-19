from multi.kmeans import kmeans_n_times_csv
from multi.dec import dec_n_times_csv
from multi.comparison import mean_rand_index
from mnist.mnist import get_mnist, subsample_mnist


import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import os
from shutil import rmtree



def test_kmeans_n_times_csv():
    Xdata = [[0,0],[0.1,0.1],[0.15,0.15],[10,10],[10.1,10.1],[10.2,10.2]]
    Ydata = [0,0,0,1,1,1]
    csv_path = "./temp/kmeans.csv"
    df_kmeans = kmeans_n_times_csv(Xdata, 3, 2, csv_path, newcsv=True, labels=Ydata)
    
    df_kmeans_loaded = pd.read_csv(csv_path,index_col=0)
    
    #Test the clustering assignment is right
    assert adjusted_rand_score(df_kmeans['kmeans_1'], [0,0,0,1,1,1]) == 1

    
    #Check they are the same from the csv
    assert np.array_equal(df_kmeans.values,df_kmeans_loaded.values)
    
    #Check columns and index are roughly right
    assert df_kmeans_loaded.index[0] == "sample_0"
    assert df_kmeans_loaded.columns[0] == "labels"
    assert df_kmeans_loaded.columns[1] == "kmeans_1"
    
    
def test_dec_n_times_csv():
    #Make some fake test images
    #Black images
    X0 = (np.zeros([3,784]) + np.random.normal(0,0.05,[3,784])).clip(min=0)
    #Grey images
    X50 = (np.ones([3,784]) * np.random.normal(0.5,0.05,[3,784])).clip(min=0,max=1)
    
    Xdata = np.concatenate([X0,X50],axis=0)
    Ydata = np.array([0,0,0,1,1,1])
    
    csv_path = "./temp/dec.csv"
    
    df_dec = dec_n_times_csv(Xdata,Ydata, 1, 2, csv_path, newcsv=True,
                finetune_iters=1000,layerwise_pretrain_iters=500,iter_max=10)
    
    df_dec_loaded = pd.read_csv(csv_path,index_col=0)
    
    #Test the clustering assignment is right
    assert adjusted_rand_score(df_dec['dec_1'], [0,0,0,1,1,1]) == 1
    
    #Check they are the same from the csv
    assert np.array_equal(df_dec.values,df_dec_loaded.values)
    
    #Check columns and index are roughly right
    assert df_dec_loaded.index[0] == "sample_0"
    assert df_dec_loaded.columns[0] == "labels"
    assert df_dec_loaded.columns[1] == "dec_1"
    
    
    
    
def test_mean_rand_index():
    Xdata = [[0,0],[0.1,0.1],[0.15,0.15],[10,10],[10.1,10.1],[10.2,10.2]]
    Ydata = [58,69,-12,6,np.inf,np.nan]
    csv_path = "./temp/kmeans2.csv"
    df_kmeans = kmeans_n_times_csv(Xdata, 10, 2, csv_path, newcsv=True, labels=Ydata)
    
    mean_rand = mean_rand_index(df_kmeans)
    #The mean rand should be 1 because you should always come up with the same
    #set of labels for this Xdata
    assert mean_rand == 1

    
# #After all tests completed, delete the temp directory
# def pytest_sessionfinish(session, exitstatus):
#     if os.path.isdir("./temp"):
#         rmtree("./temp")
        
def test_tidy():
    try:
        rmtree("./temp")
    except:
        raise Exception('Failed to delete temp folder in test_multi')
                     
    
    