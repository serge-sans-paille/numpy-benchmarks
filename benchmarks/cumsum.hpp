#from: http://stackoverflow.com/questions/16541618/perform-a-reverse-cumulative-sum-on-a-numpy-array
#pythran export cumsum(float[])
#setup: import numpy as np ; r = np.random.rand(1000000)
#run: cumsum(r)

import numpy as np
def cumsum(x):
    return np.cumsum(x)
