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

template<typename X, typename Y>
auto lstsqr(X const& x, Y const& y)
{
  auto x_avg = mean(x)();
  auto y_avg = mean(y)();
  auto dx = eval(x - x_avg);
  auto dy = y - y_avg;
  auto var_x = sum(square(dx))();
  auto cov_xy = sum(dx * dy)();
  auto slope = cov_xy / var_x;
  auto y_interc = y_avg - slope * x_avg;
  return std::make_tuple(slope, y_interc);
}

std::tuple<double, double>
py_lstsqr(pytensor<double, 1> const& x, pytensor<double, 1> const& y)
{
  return lstsqr(x, y);
}

PYBIND11_MODULE(xtensor_lstsqr, m)
{
    import_numpy();
    m.def("lstsqr", py_lstsqr);
}
