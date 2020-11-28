#from: https://github.com/serge-sans-paille/pythran/issues/1668
#setup: import numpy as np; s3 = np.ones((2, 3 * 10**3), dtype=np.complex128); wxy0 = np.zeros((2,2,21), dtype=np.complex128); wxy0[0][0][21//2] = wxy0[1][1][21//2] = 1
#run: cma_pt(s3, wxy0, 1e-3, 1., 2)

#pythran export cma_pt(complex128[][], complex128[][][], float64, float64, int)
import numpy as np

def apply_filter_pt(E, wx):
    Xest = np.sum(E*np.conj(wx))
    return Xest

def cma_pt(E, wxy, mu, R, os):
    pols, L = E.shape
    assert wxy.shape[0] == pols
    assert wxy.shape[1] == pols
    ntaps = wxy.shape[-1]
    N = (L//os//ntaps-1)*ntaps
    err = np.zeros((pols, L), dtype=np.complex128)
    for k in range(pols):
        for i in range(N):
            X = E[:, i*os:i*os+ntaps]
            Xest = apply_filter_pt(X, wxy[k])
            err[k,i] = (R-abs(Xest)**2)*Xest
            wxy[k] += mu*np.conj(err[k,i])*X
    return err, wxy
