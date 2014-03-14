#setup: import numpy as np ; N = 100 ; X, Y = np.random.randn(100,N), np.random.randn(20,N)
#run: allpairs_distances_loops(X, Y)

#pythran export allpairs_distances_loops(float64[][], float64[][])
import numpy as np

def allpairs_distances_loops(X,Y):
  result = np.zeros( (X.shape[0], Y.shape[0]), X.dtype)
  for i in xrange(X.shape[0]):
    for j in xrange(Y.shape[0]):
      result[i,j] = np.sum( (X[i,:] - Y[j,:]) ** 2)
  return result 
