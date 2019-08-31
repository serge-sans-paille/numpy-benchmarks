import numpy as np
#setup: N=100
#run: mandelbrot(0., 100., 0., 100., 100., 100., N, 50.)
#pythran export mandelbrot(float, float, float, float, float, float, int, float)
def mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon):
    X = np.linspace(xmin, xmax, int(xn))
    Y = np.linspace(ymin, ymax, int(yn))
    C = X + Y[:, None]*1j
    N = np.zeros(C.shape, dtype=np.int64)
    Z = np.zeros(C.shape, np.complex128)
    for n in range(maxiter):
        I = np.less(np.abs(Z), horizon)
        N[I] = n
        Z[I] = Z[I]**2 + C[I]
    N[N == maxiter-1] = 0
    return Z, N

