import numpy as np
import GPflow
#from scipy.cluster.vq import kmeans, kmeans2, whiten
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
#import hdbscan #results are not good with hdbscan
import copy
import sys


class svDGPOM():
   def __init__(self, M=50, max_iters_for_opt=10, filter_informative_data=True, clustering_method='DBSCAN',\
                optimization_method='L-BFGS-B', initial_scan=True, filter_informative_data_thresh=5, n_clusters=50, bring_forward_paras=True):
      self.M = M #50 #100 #200 #50 is okay
      self.max_iters_for_opt = max_iters_for_opt #10
      self.filter_informative_data = filter_informative_data #True
      self.clustering_method = clustering_method #'DBSCAN'
      self.optimization_method = optimization_method #'L-BFGS-B' #tf.train.AdagradOptimizer(learning_rate=0.3) #tf.train.AdamOptimizer(learning_rate=0.3)
      self.initial_scan = initial_scan
      self.filter_informative_data_thresh = filter_informative_data_thresh
      self.n_clusters = n_clusters
      self.bring_forward_paras = bring_forward_paras
      self.nDim = 2

   def kernel(self):
       kern = GPflow.kernels.RBF(self.nDim, ARD=False) + GPflow.kernels.White(self.nDim)
       kern.white.variance = 0.01
       return kern

   def likelihood(self):
       return GPflow.likelihoods.Bernoulli()

   def learn(self,Xtrain, Ytrain):

      if self.initial_scan is True: #the first scan
         self.initial_scan = False

         if self.clustering_method == 'random':
            Z = Xtrain[np.random.choice(Xtrain.shape[0], self.M, replace=False), :]
         else:
            #kmeans for selecting Z
            est = KMeans(n_clusters=np.int16(self.M/2))
            Z_0 = est.fit(Xtrain[Ytrain[:,0]==0, :]).cluster_centers_
            Z_1 = est.fit(Xtrain[Ytrain[:,0]==1, :]).cluster_centers_
            Z = np.vstack((Z_0, Z_1))

         self.mdl = GPflow.svgp.SVGP(Xtrain, Ytrain, kern=self.kernel(), likelihood=self.likelihood(), Z=Z) #minibatch_size=20

         self.mdl.optimize(max_iters=100, method='L-BFGS-B') #100

      else:#not the first scan
         #if we need to filter informative points
         if self.filter_informative_data_thresh >= 0:
            query = self.mdl.predict_y(Xtrain)
            query_mean = query[0]
            query_var = query[1]
            mahal_dist = np.absolute((Ytrain - query_mean)/query_var).ravel()

            informative_points = mahal_dist > self.filter_informative_data_thresh #thresholded_mahal_dist

            if np.sum(informative_points) == 0:
               print('This scan has no information! Let\'s grab the next data frame.')
               return
         else:
            informative_points = [True]*Xtrain.shape[0] #otherwise, all are informatice points

         #let's do clustering to choose Z
         Z = self._choose_Z(Xtrain, Ytrain, informative_points)

         #stack new data
         X_stacked = np.vstack((self.mdl.X._array, Xtrain))
         Y_stacked = np.vstack((self.mdl.Y._array, Ytrain))
         Z_stacked = np.vstack((self.mdl.Z._array, Z))
         print('m={} and M={}'.format(Z.shape[0], Z_stacked.shape[0]))

         #note: do not aggregate data/paras, if you do not want the long-term map (i.e. average over a period)

         #get parameters that have already been optimized
         kern_rbf_lengthscales_array = copy.copy(self.mdl.kern.rbf.lengthscales._array)
         kern_rbf_variance_array = copy.copy(self.mdl.kern.rbf.variance._array)
         kern_white_variance_array = copy.copy(self.mdl.kern.white.variance._array)
         q_mean = copy.copy(self.mdl.q_mu._array)
         q_sqrt = copy.copy(self.mdl.q_sqrt._array)

         #re-instantiate the model with new X, Y, Z
         #this is done due to limitations of GPflow. the statndalone bigdataGP can skip this)
         self.mdl = GPflow.svgp.SVGP(X_stacked, Y_stacked, kern=self.kernel(), likelihood=self.likelihood(), Z=Z_stacked) #minibatch_size=20

         #set hyper/-parameters that have already been optimized - so that we do not need to optimize everything again
         if self.bring_forward_paras is True:
            self.mdl.kern.rbf.lengthscales._array = kern_rbf_lengthscales_array
            self.mdl.kern.rbf.variance._array = kern_rbf_variance_array
            self.mdl.kern.white.variance._array = kern_white_variance_array
            self.mdl.q_mu._array[:q_mean.shape[0], :] = q_mean
            if self.nDim == 2:
               self.mdl.q_sqrt._array[:q_sqrt.shape[0], :q_sqrt.shape[1], :] = q_sqrt
            elif self.nDim == 3:
               self.mdl.q_sqrt._array[:q_sqrt.shape[0], :q_sqrt.shape[1], :q_sqrt.shape[2]] = q_sqrt

         #optimize all hyper/-parameters again
         self.mdl.optimize(max_iters=self.max_iters_for_opt, method=self.optimization_method)

      self.m = Z.shape[0]

   def predict(self, Xtest):
      return self.mdl.predict_y(Xtest)

   def _choose_Z(self, Xtrain, Ytrain, informative_points):
      #extract informative points for choosing inducing points
      Xtrain_informative = Xtrain[informative_points, :]
      Ytrain_informative = Ytrain[informative_points, :]

      #let's do clusteing
      if self.clustering_method == 'random':
         rand_indx = np.random.choice(Xtrain_informative.shape[0], self.M, replace=False)
         Z = Xtrain_informative[rand_indx, :]
      elif self.clustering_method == 'k-means':
         est = KMeans(n_clusters = self.n_clusters)
         Z_0 = est.fit(Xtrain_informative[Ytrain_informative[:,0]==0, :]).cluster_centers_
         Z_1 = est.fit(Xtrain_informative[Ytrain_informative[:,0]==1, :]).cluster_centers_
         Z = np.vstack((Z_0, Z_1))
      elif self.clustering_method == 'DBSCAN':
         est = DBSCAN(eps=3, min_samples=1) #est = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1) #HDBSCAN clustering
         cluster_labels = est.fit_predict(Xtrain_informative) #real clusters 0, 1, 2... noise points = -1
         n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
         for ith_cluster_label in range(n_clusters): #find the centroid for each real cluster
            cluster_centroid = np.average(Xtrain_informative[cluster_labels == ith_cluster_label, :], axis=0)
            if ith_cluster_label == 0:
               Z = cluster_centroid
            else:
               Z = np.vstack((Z, cluster_centroid))
         if -1 in cluster_labels: #make noise points clusters
            cluster_centroid = Xtrain_informative[cluster_labels == -1, :]
            if 'Z' not in locals():#if no clusters, but all noise ponts
               Z = cluster_centroid
            else: #stack cluster centroids and all noise points
               Z = np.vstack((Z, cluster_centroid))
      elif self.clustering_method == 'affinity_propagation':
         est = AffinityPropagation()
         Z = est.fit(Xtrain_informative).cluster_centers_
      else:
         print('My Err - Invalid clustering method!')
         sys.exit()

      return Z
