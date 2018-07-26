#setup: import numpy as np, random; np.random.seed(0); s=np.random.randn(2**16)+np.random.randn(2**16)*1.j ; sc=np.random.choice(s, 32)
#run: make_decision(s, sc)

import numpy as np
#pythran export make_decision(complex128[], complex128[])
def make_decision(E, symbols):
    L = E.shape[0]
    syms_out = np.zeros(L, dtype=E.dtype)
    for i in range(L):
        im = np.argmin(abs(E[i]-symbols)**2)
        syms_out[i] = symbols[im]
    return syms_out
