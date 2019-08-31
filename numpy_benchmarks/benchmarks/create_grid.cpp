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

template<typename X>
auto create_grid(X const& x)
{
  xarray<double> x_t = x;
  auto N = x_t.shape()[0];
  using shape_type = decltype(N);
  xarray<double> z = zeros<double>({N, N, (shape_type)3});
  view(z, all(), all(), 0) = x_t.reshape({N , (shape_type)1});
  view(z, all(), all(), 1) = x_t.reshape({N});
  auto fast_grid = z.reshape({N * N, (shape_type)3});
  return fast_grid;
}

pytensor<double, 2> py_create_grid(pytensor<double, 1> const& x)
{
  return create_grid(x);
}

PYBIND11_MODULE(xtensor_create_grid, m)
{
    import_numpy();
    m.def("create_grid", py_create_grid);
}
