#setup: import numpy as np ; N = 100 ; X, Y = np.random.randn(100,N), np.random.randn(20,N)
#run: allpairs_distances(X, Y)

#pythran export allpairs_distances(float64[][], float64[][])
import numpy as np

def allpairs_distances(X,Y):
  return np.array([[np.sum( (x-y) ** 2) for x in X] for y in Y])
