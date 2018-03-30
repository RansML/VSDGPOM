import sys
import numpy as np
from sklearn import metrics
import csv
import argparse
import util2

def read_txy_csv(fn):
   data = readCsvFile(fn)
   Xtest = data[:, :3]
   Ytest = data[:, 3][:, np.newaxis]
   return Xtest, Ytest

def read_carmen(fn_gfs):
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--logfile",
            default=fn_gfs, #wombot_test_2016-02-05-11-51-14.gfs.log  SimonFewLasers.gfs.log
            help="Logfile in CARMEN format to process"
    )

    args = parser.parse_args()

    # Load data and split it into training and testing data
    train_data, test_data = util2.create_test_train_split(args.logfile, 0)

    train_data = np.asarray( train_data["scans"])

    return train_data

def get_mesh_grid(resolution=1, limits=[-120, 120, -20, 120]):
   x_spaced = np.arange(limits[0], limits[1], resolution )
   y_spaced = np.arange(limits[2], limits[3], resolution)
   xx, yy = np.meshgrid(x_spaced, y_spaced)
   X_plot = np.vstack((xx.flatten(),yy.flatten())).T
   return X_plot

def generate_test_data_everyother(fn, X_all, Y_all, step=2):
    #temp - rand method
   #ith_scan_indx_plus1 = X_all[:, 0] > ith_scan
   #Xtest = X_all[ith_scan_indx_plus1, 1:]
   #Ytest = Y_all[ith_scan_indx_plus1, :]

   for jth_scan in range(1, len(np.unique(X_all[:, 0])), step): #start from 1

      jth_scan_indx = X_all[:, 0] == jth_scan

      X_jth_scan = X_all[jth_scan_indx, :]
      Y_jth_scan = Y_all[jth_scan_indx, :]

      X_jth_0 = X_jth_scan[Y_jth_scan[:,0]==0, :]
      X_jth_1 = X_jth_scan[Y_jth_scan[:,0]==1, :]
      if X_jth_1.shape[0] < X_jth_0.shape[0]: #number of ones are less than number of zeros
          n_X_jth_1 = X_jth_1.shape[0]
          indx = np.random.choice(X_jth_0.shape[0], n_X_jth_1, replace=False)
          X_jth_0 = X_jth_0[indx, :]
          n_X_jth_0 = n_X_jth_1
      else:
          n_X_jth_0 = X_jth_0.shape[0]
          indx = np.random.choice(X_jth_1.shape[0], n_X_jth_0, replace=False)
          X_jth_1 = X_jth_1[indx, :]
          n_X_jth_1 = n_X_jth_0
          print('ouch!', jth_scan)

      X_jth = np.vstack((X_jth_0, X_jth_1))
      Y_jth = np.array([[0]*n_X_jth_0 + [1]*n_X_jth_1]).T

      if jth_scan == 1: #first scan to store
         Xtest = X_jth
         Ytest = Y_jth
      else:
         Xtest = np.vstack((Xtest, X_jth))
         Ytest = np.vstack((Ytest, Y_jth))

   points_labels = np.hstack((Xtest, Ytest))
   np.savetxt(fn, points_labels, delimiter=",")

def readCsvFile( fileName ):
    reader = csv.reader(open(fileName,'r') )
    dataList = []
    for row in reader:
        dataList.append( [float(elem) for elem in row ] )
    return np.array( dataList )

