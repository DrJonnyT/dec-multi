#Set tensorflow gpu memory to be able to grow, which makes it less likely to crash
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


from multi.kmeans import kmeans_n_times_csv, kmeans_mnist_n_times, kmeans_mnist_n_times_csv
from multi.dec import dec_n_times_csv, dec_mnist_n_times_csv



import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from shutil import rmtree
import pytest
import os
import tensorflow as tf
import subprocess



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
    assert df_dec_loaded.shape == (6,3)
    
    

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
    indices_path = "./temp/dec_mnist_indices.csv"
    
    #try:
    df_dec, df_labels, df_indices = dec_mnist_n_times_csv(100, 2, 10,csv_path,overwrite=True,
                    finetune_iters=1000,layerwise_pretrain_iters=500,iter_max=10,
                    verbose=0)
    # except:
    #     raise Exception("The system has probably run out of VRAM") 
    
    assert np.shape(df_dec) == (100,2)
    assert np.shape(df_labels) == (100,2)
    
    assert df_dec.index[0] == "sample_0"
    assert df_dec.columns[0] == "dec_1"
    assert df_labels.columns[0] == "labels_1"
    
    assert np.array_equal(np.unique(df_dec['dec_1']), [0,1,2,3,4,5,6,7,8,9])
    #Check it's not 10 of each digit in a row (what you would get from balanced)
    assert not np.array_equal(df_labels['labels_1'], np.repeat([0,1,2,3,4,5,6,7,8,9],10))
    
    #Check they are the same from the csv
    df_dec_loaded = pd.read_csv(csv_path,index_col=0)
    df_labels_loaded = pd.read_csv(labels_path,index_col=0)
    df_indices_loaded = pd.read_csv(indices_path,index_col=0)
    assert np.array_equal(df_dec.values,df_dec_loaded.values)
    assert np.array_equal(df_labels.values,df_labels_loaded.values)
    assert np.array_equal(df_indices.values,df_indices_loaded.values)
    
    
    
    #Now append another 2 runs to this preexisting csv and test
    #2 + 2 = 4 runs in total
    #These 2 extra runs are balanced, so 10 of each digit
    df_dec, df_labels, df_indices = dec_mnist_n_times_csv(100, 4, 10,csv_path,overwrite=False,
                    finetune_iters=1000,layerwise_pretrain_iters=500,iter_max=10,
                    verbose=0,balanced=True)
    
    assert np.shape(df_dec) == (100,4)
    assert np.shape(df_labels) == (100,4)

    assert df_dec.index[0] == "sample_0"
    assert df_dec.columns[0] == "dec_1"
    assert df_labels.columns[0] == "labels_1"
    assert df_dec.columns[-1] == "dec_4"
    assert df_labels.columns[-1] == "labels_4"

    assert np.array_equal(np.unique(df_dec['dec_4']), [0,1,2,3,4,5,6,7,8,9])
    #10 of each digit
    assert np.array_equal(df_labels['labels_4'], np.repeat([0,1,2,3,4,5,6,7,8,9],10))


    #Test with too many mnist digits
    with pytest.raises(Exception) as e_info:
        dec_mnist_n_times_csv(63140, 1, 10,csv_path,newcsv=False,
                        finetune_iters=1000,layerwise_pretrain_iters=500,iter_max=10,
                        verbose=0,balanced=True)
    with pytest.raises(Exception) as e_info:
        dec_mnist_n_times_csv(1e6, 1, 10,csv_path,newcsv=False,
                        finetune_iters=1000,layerwise_pretrain_iters=500,iter_max=10,
                        verbose=0,balanced=False)
        



def get_total_vram():
    """
    Work out the total VRAM of the GPU in the system

    Returns
    -------
    total_vram : int
        Total VRAM (MB).

    """
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'])
        total_vram = int(output.strip())
        return total_vram
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Failed to retrieve total VRAM.")
        return 0


#Same as test_dec_mnist_n_times_csv but with full mnist dataset
#Will fail if you don't have enough VRAM
@pytest.mark.skipif(get_total_vram() < 6144,
                    reason="Insufficient VRAM for the test_dec_mnist_n_times_csv_full.")
def test_dec_mnist_n_times_csv_full():
    csv_path = "./temp/dec_mnist_full.csv"
    labels_path = "./temp/dec_mnist_full_labels.csv"

    #Test with the full mnist dataset
    tf.get_logger().setLevel('FATAL')
    dec_mnist_n_times_csv(0, 1, 10,csv_path,overwrite=True,
                    finetune_iters=1000,layerwise_pretrain_iters=500,iter_max=10,
                    verbose=0)
    #Check it made a file with the right number of samples
    df_dec = pd.read_csv(csv_path,index_col=0)
    df_labels = pd.read_csv(labels_path,index_col=0)
    assert df_dec.shape[0] == 70000
    assert df_labels.shape[0] == 70000




    
#After all tests completed, delete the temp directory
def pytest_sessionfinish(session, exitstatus):
    if os.path.isdir("./temp"):
        rmtree("./temp")
        
def test_tidy():
    try:
        rmtree("./temp")
    except:
        raise Exception('Failed to delete temp folder in test_multi')
                     
    
    