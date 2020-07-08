#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xindex_view.hpp"

#define FORCE_IMPORT_ARRAY

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

#include "pybind11/stl.h"


using namespace xt;

auto mandelbrot(double xmin, double xmax, double ymin, double ymax, double xn, double yn, long maxiter, double horizon)
{
    auto X = linspace<double>(xmin, xmax, int(xn));
    auto Y = linspace<double>(ymin, ymax, int(yn));
    auto C = eval(X + view(Y, all(), newaxis())*std::complex<double>(0, 1));
    xtensor<int64_t, 2> N = zeros<int64_t>(C.shape());
    xtensor<std::complex<double>, 2> Z = zeros<std::complex<double>>(C.shape());
    for(long n = 0; n < maxiter; ++n) {
        auto I = eval(abs(Z) < horizon);
        filter(N, I) = n;
        filter(Z, I) = filter(Z, I)*filter(Z, I) + filter(C, I);
    }
    filter(N, equal(N, maxiter-1)) = 0;
    return std::make_tuple(Z, N);

}

std::tuple<pytensor<std::complex<double>, 2>, pytensor<int64_t, 2>> py_mandelbrot(double xmin, double xmax, double ymin, double ymax, double xn, double yn, long maxiter, double horizon)
{
  return mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon);
}

PYBIND11_MODULE(xtensor_mandelbrot, m)
{
    import_numpy();
    m.def("mandelbrot", py_mandelbrot);
}


