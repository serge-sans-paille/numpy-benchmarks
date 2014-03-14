#setup: import numpy as np ; a = np.arange(0,1,10e-5)
#run: smoothing(a, .4)
#from: http://www.parakeetpython.com/

#pythran export smoothing(float[], float)

def smoothing(x, alpha):
  """
  Exponential smoothing of a time series
  For x = 10**6 floats
  - Python runtime: 9 seconds
  - Parakeet runtime: .01 seconds
  """
  s = x.copy()
  for i in xrange(1, len(x)):
    s[i] = alpha * x[i] + (1 - alpha) * s[i-1]
  return s
