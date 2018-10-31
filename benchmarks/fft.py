#setup: N = 2**10 ; import numpy as np ; np.random.seed(0); a = np.array(np.random.rand(N), dtype=complex)
#run: fft(a)

#pythran export fft(complex [])

import math, numpy as np

def fft(x):
   N = x.shape[0]
   if N == 1:
       return np.array(x)
   e=fft(x[::2])
   o=fft(x[1::2])
   M=N//2
   l=e + o * math.e**(-2j*math.pi*np.arange(M)/N)
   r=e - o * math.e**(-2j*math.pi*np.arange(M)/N)
   return np.array(l+r)

