from __future__ import division

#import sys
import numpy as np
import util
#from DOGM import DynamicOccupancyGridmap as DOGM
#import DGPOM
#import time as comp_timer
#import svDGPOM
#import plotting_methods
import matplotlib.pyplot as pl
import csv

def readCsvFile( fileName ):
    reader = csv.reader(open(fileName,'r') )
    dataList = []
    for row in reader:
        dataList.append( [float(elem) for elem in row[:] ] )
    return np.array( dataList )

def main():
    for i in range(0,24,2):
        print i
        grid = readCsvFile('outputs/vitor/grid.csv')
        map = readCsvFile('outputs/vitor/map_{}.csv'.format(i))
        #print(grid, map, map.min(), map.max())
        indx = np.logical_and(map.ravel()>0.8, map.ravel()>0.8)
        map_surface = grid[indx,:]
        np.savetxt('outputs/vitor/grid_surface_{}.xyz'.format(i), map_surface, delimiter=' ')

if __name__ == '__main__':
    main()
