#from: http://people.duke.edu/~ccc14/sta-663-2016/03A_Numbers.html#Example:-Calculating-pairwise-distance-matrix-using-broadcasting-and-vectorization
#setup: import numpy as np ; X = np.linspace(0,10,20000).reshape(200,100)
#run: pairwise(X)

#pythran export pairwise(float [][])

import numpy as np
def pairwise(pts):
    return np.sum((pts[None,:] - pts[:, None])**2, -1)**0.5
