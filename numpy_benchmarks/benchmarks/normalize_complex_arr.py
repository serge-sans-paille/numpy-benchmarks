import numpy as np
#from: https://stackoverflow.com/questions/41576536/normalizing-complex-values-in-numpy-python
#setup: import numpy as np; np.random.seed(0); N = 10000; x = np.random.random(N) + 1j *  np.random.random(N)
#run: normalize_complex_arr(x)

#pythran export normalize_complex_arr(complex[])

def normalize_complex_arr(a):
    a_oo = a - a.real.min() - 1j*a.imag.min() # origin offsetted
    return a_oo/np.abs(a_oo).max()
