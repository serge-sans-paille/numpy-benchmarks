#from: https://stackoverflow.com/questions/55854611/efficient-way-of-vectorizing-distance-calculation/55877642#55877642
#setup: import numpy as np ; np.random.seed(0); N = 80; x = np.random.rand(N,N); y = np.random.rand(N,N)
#run: l1norm(x, y)

#pythran export l1norm(float64[][], float64[:,:])
import numpy as np
def l1norm(x, y):
    return np.sum(np.abs(x[:, None, :] - y), axis=-1)
