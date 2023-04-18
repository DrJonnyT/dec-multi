from multi.kmeans import kmeans_n_times_csv
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import os



def test_kmeans_n_times_csv():
    Xdata = [[0,0],[0.1,0.1],[0.15,0.15],[10,10],[10.1,10.1],[10.2,10.2]]
    Ydata = [0,0,0,1,1,1]
    csv_path = "./temp/temp.csv"
    df_kmeans = kmeans_n_times_csv(Xdata, 3, 2, csv_path, newcsv=True, labels=Ydata)
    
    df_kmeans_loaded = pd.read_csv(csv_path,index_col=0)
    
    #Test the clustering assignment is right
    assert adjusted_rand_score(df_kmeans['kmeans_1'], [0,0,0,1,1,1]) == 1

    
    #Check they are the same from the csv
    np.array_equal(df_kmeans.values,df_kmeans_loaded.values)
    
    #Check columns and index are roughly right
    assert df_kmeans_loaded.index[0] == "sample_0"
    assert df_kmeans_loaded.columns[0] == "labels"
    assert df_kmeans_loaded.columns[1] == "kmeans_1"
    
    #Tidy up temp files
    os.remove(csv_path)
    os.rmdir("./temp")
    
                          
                          
    
    