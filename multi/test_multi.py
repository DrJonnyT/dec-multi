from multi.kmeans import kmeans_n_times_csv, kmeans_mnist_n_times, kmeans_mnist_n_times_csv
from multi.dec import dec_n_times_csv, dec_mnist_n_times_csv
from multi.comparison import mean_rand_index


import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from shutil import rmtree
import pytest



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
                finetune_iters=1000,layerwise_pretrain_iters=500,iter_max=10,
                verbose=0)
    
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




def test_kmeans_mnist_n_times():
    df_kmeans, df_kmeans_labels = kmeans_mnist_n_times(10, 5, 10)
    
    assert np.shape(df_kmeans) == (100,5)
    assert np.shape(df_kmeans_labels) == (100,5)
    
    assert df_kmeans.index[0] == "sample_0"
    assert df_kmeans.columns[0] == "kmeans_1"
    assert df_kmeans_labels.columns[0] == "labels_1"
    
    assert np.array_equal(np.unique(df_kmeans['kmeans_1']),[0,1,2,3,4,5,6,7,8,9])
    
    #Test with too many mnist digits
    with pytest.raises(Exception) as e_info:
        kmeans_mnist_n_times(6314, 5, 10)



def test_kmeans_mnist_n_times_csv():
    csv_path = "./temp/kmeans3.csv"
    kmeans_mnist_n_times_csv(10, 5, 10,csv_path)
    
    df_kmeans = pd.read_csv(csv_path,index_col=0)
    df_labels = pd.read_csv("./temp/kmeans3_labels.csv",index_col=0)
    
    assert np.shape(df_kmeans) == (100,5)
    assert np.shape(df_labels) == (100,5)
    
    assert df_kmeans.index[0] == "sample_0"
    assert df_kmeans.columns[0] == "kmeans_1"
    assert df_labels.columns[0] == "labels_1"
    
    assert np.array_equal(np.unique(df_kmeans['kmeans_1']),[0,1,2,3,4,5,6,7,8,9])
    assert np.array_equal(df_labels['labels_1'], np.repeat([0,1,2,3,4,5,6,7,8,9],10))

    #Test with too many mnist digits
    with pytest.raises(Exception) as e_info:
        kmeans_mnist_n_times_csv(6314, 1, 10,csv_path)


def test_dec_mnist_n_times_csv():
    csv_path = "./temp/dec_mnist.csv"
    labels_path = "./temp/dec_mnist_labels.csv"
    df_dec, df_labels = dec_mnist_n_times_csv(10, 2, 10,csv_path,newcsv=True,
                    finetune_iters=1000,layerwise_pretrain_iters=500,iter_max=10,
                    verbose=0)
    
    assert np.shape(df_dec) == (100,2)
    assert np.shape(df_labels) == (100,2)
    
    assert df_dec.index[0] == "sample_0"
    assert df_dec.columns[0] == "dec_1"
    assert df_labels.columns[0] == "labels_1"
    
    assert np.array_equal(np.unique(df_dec['dec_1']), [0,1,2,3,4,5,6,7,8,9])
    assert np.array_equal(df_labels['labels_1'], np.repeat([0,1,2,3,4,5,6,7,8,9],10))
    
    
    #Check they are the same from the csv
    df_dec_loaded = pd.read_csv(csv_path,index_col=0)
    df_labels_loaded = pd.read_csv(labels_path,index_col=0)
    assert np.array_equal(df_dec.values,df_dec_loaded.values)
    assert np.array_equal(df_labels.values,df_labels_loaded.values)
    
    
    
    #Now append another 2 runs to this preexisting csv and test
    #2 + 2 = 4 runs in total
    df_dec, df_labels = dec_mnist_n_times_csv(10, 4, 10,csv_path,newcsv=False,
                    finetune_iters=1000,layerwise_pretrain_iters=500,iter_max=10,
                    verbose=0)
    
    assert np.shape(df_dec) == (100,4)
    assert np.shape(df_labels) == (100,4)

    assert df_dec.index[0] == "sample_0"
    assert df_dec.columns[0] == "dec_1"
    assert df_labels.columns[0] == "labels_1"
    assert df_dec.columns[-1] == "dec_4"
    assert df_labels.columns[-1] == "labels_4"

    assert np.array_equal(np.unique(df_dec['dec_4']), [0,1,2,3,4,5,6,7,8,9])
    assert np.array_equal(df_labels['labels_4'], np.repeat([0,1,2,3,4,5,6,7,8,9],10))


    #Test with too many mnist digits
    with pytest.raises(Exception) as e_info:
        dec_mnist_n_times_csv(6314, 1, 10,csv_path,newcsv=False,
                        finetune_iters=1000,layerwise_pretrain_iters=500,iter_max=10,
                        verbose=0)



    
# #After all tests completed, delete the temp directory
# def pytest_sessionfinish(session, exitstatus):
#     if os.path.isdir("./temp"):
#         rmtree("./temp")
        
def test_tidy():
    try:
        rmtree("./temp")
    except:
        raise Exception('Failed to delete temp folder in test_multi')
                     
    
    