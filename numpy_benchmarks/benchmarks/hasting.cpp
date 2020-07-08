#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xfixed.hpp"
#define FORCE_IMPORT_ARRAY

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"


using namespace xt;

template<typename Y, typename T, typename A1, typename A2, typename B1, typename B2, typename D1, typename D2>
auto hasting(Y const& y, T const& t, A1 const& a1, A2 const& a2, B1 const& b1, B2 const& b2, D1 const& d1, D2 const& d2)
{
  xtensor<double, 1> yprime = empty<double>({3});
  yprime[0] = y[0] * (1. - y[0]) - a1*y[0]*y[1]/(1. + b1 * y[0]);
  yprime[1] = a1*y[0]*y[1] / (1. + b1 * y[0]) - a2 * y[1]*y[2] / (1. + b2 * y[1]) - d1 * y[1];
  yprime[2] = a2*y[1]*y[2]/(1. + b2*y[1]) - d2*y[2];
  return yprime;
}

pytensor<double, 1> py_hasting(pytensor<double, 1> const& y, double t, double a1, double a2, double b1, double b2, double d1, double d2)
{
  return hasting(y, t, a1, a2, b1, b2, d1, d2);
}

PYBIND11_MODULE(xtensor_hasting, m)
{
    import_numpy();
    m.def("hasting", py_hasting);
}
