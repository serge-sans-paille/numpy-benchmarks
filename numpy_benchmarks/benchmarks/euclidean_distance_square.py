#from:  https://stackoverflow.com/questions/50658884/why-this-numba-code-is-6x-slower-than-numpy-code
#setup: import numpy as np; np.random.seed(0); x1 = np.random.random((1, 512)); x2 = np.random.random((10000, 512))
#run: euclidean_distance_square(x1, x2)
#pythran export euclidean_distance_square(float64[1,:], float64[:,:])
import numpy as np
def euclidean_distance_square(x1, x2):
    return -2*np.dot(x1, x2.T) + np.sum(np.square(x1), axis=1)[:, np.newaxis] + np.sum(np.square(x2), axis=1)
