################################
#Required:
#python 2.7
#numpy
#matplotlib
#sklearn
#GPflow 0.3.5
#################################

from __future__ import division

import numpy as np
import matplotlib.pyplot as pl
import util
import svDGPOM
from mayavi import mlab
import marching_cubes

def load_parameters(case):
    #train_all, test, gfs, max_laser_distance, limits

    parameters = \
        {'sim': \
             ('data/icra17/icra17_sim_filled.csv',\
              49,\
              [-120, 120, -20, 120]),
         'vitor': \
             ('data/ijrr_vitor/kitti1.csv', \
              40, \
              [-15, +15, -15, +15, -5, 10]),
        }

    return parameters[case]

def main():
      #load parameters
      fn_train, laser_max_distance, lims = load_parameters('sim')

      #load data
      #data is assumed to be already sampled from lidar beams as unoccupied and free
      #col0=scan index; col1=longitude; col2=latitude; col3=occupied/unoccupied
      X_all, Y_all = util.read_txy_csv(fn_train)
      X_plot2, X_plot2_shape = util.get_mesh_grid(resolution=0.5, limits=lims)
      X_plot = np.copy(X_plot2)
      if len(lims) == 6:
          X_plot[:, 0] = X_plot2[:, 1]
          X_plot[:, 1] = X_plot2[:, 0]
          X_plot[:, 2] = X_plot2[:, 2]

      #define the model
      model_svDGPOM = svDGPOM.svDGPOM(M=50, filter_informative_data_thresh=5) #M  = number of inducing points to kick it off with

      max_t = len(np.unique(X_all[:, 0]))
      for ith_scan in range(0, max_t, 2): #every other

         #extract data points of the ith scan
         ith_scan_indx = X_all[:, 0] == ith_scan
         print('\n{}th scan: size={}'.format(ith_scan, np.sum(ith_scan_indx)))

         Xtrain_ith_scan = X_all[ith_scan_indx, 1:]
         Ytrain_ith_scan = Y_all[ith_scan_indx, :]

         #let's train
         model_svDGPOM.learn(Xtrain_ith_scan, Ytrain_ith_scan)
         print('Size of X = {} data points'.format(model_svDGPOM.mdl.X.value.shape[0]))

         #predictions
         qry_svDGPOM = model_svDGPOM.predict(X_plot)
         qry_svDGPOM_mean = qry_svDGPOM[0] #mean prediction

         """
         map = qry_svDGPOM_mean.ravel()
         np.savetxt('outputs/vitor/map_' + str(ith_scan) + '.csv', map, delimiter=',')
         continue

         fig1 = mlab.figure(size=(1800, 1000))
         s = mlab.points3d(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2], qry_svDGPOM_mean.ravel(), mode='sphere', scale_factor=3E-2, scale_mode='none')
         s.glyph.glyph.clamping = False
         mlab.orientation_axes()
         mlab.axes(nb_labels=3)
         mlab.colorbar(s, title='', orientation='vertical', nb_labels=5, label_fmt='%.1f')
         mlab.points3d(0, 0, 0, scale_factor=4E-2)  # origin
         mlab.show()

         yq = np.reshape(qry_svDGPOM_mean, X_plot2_shape)
         # Use marching cubes to obtain the surface mesh of these ellipsoids
         verts, faces = marching_cubes.marching_cubes_classic(yq, 0.5)  # alternative: #import mcubes or from skimage import measure
         print(verts.shape, faces.shape)
         fig1 = mlab.figure(size=(1800, 1000))
         # s = mlab.points3d(Xq[:,0], Xq[:,1], Xq[:,2], yq.ravel(), mode='sphere', scale_factor=4E-3, scale_mode='none')
         s = mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces, opacity=1.0)  # , colormap='gray'
         # mlab.points3d(X[:,0], X[:,1], X[:,2], y.ravel(), mode='sphere', scale_factor=4E-1, scale_mode='none')
         # s.glyph.glyph.clamping = False
         mlab.show()
         """

         #plot
         pl.figure(figsize=(10,4))
         pl.subplot(121)
         pl.scatter(Xtrain_ith_scan[:,0][Ytrain_ith_scan[:,0]==1], Xtrain_ith_scan[:,1][Ytrain_ith_scan[:,0]==1], color='r', marker='.', alpha=0.5, label='Occupied')
         pl.scatter(Xtrain_ith_scan[:,0][Ytrain_ith_scan[:,0]==0], Xtrain_ith_scan[:,1][Ytrain_ith_scan[:,0]==0], color='b', marker='.', alpha=0.5, label='Free')
         pl.scatter(model_svDGPOM.mdl.Z.value[:, 0], model_svDGPOM.mdl.Z.value[:, 1], color='k', marker='x', alpha=0.7, label='Inducing')
         pl.legend()
         pl.title('Data | iter={}'.format(ith_scan))
         pl.subplot(122)
         X_plot
         pl.scatter(X_plot[:, 0], X_plot[:, 1], c=qry_svDGPOM_mean.ravel(), s=12, cmap='jet', vmin=0, vmax=1, edgecolor='');
         pl.colorbar()
         pl.title('VSDGPOM | iter={}'.format(ith_scan))
         pl.show()
         #pl.savefig('outputs/sim3d_{}.png'.format(ith_scan))

if __name__ == '__main__':
    main()

