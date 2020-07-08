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

template<typename Array>
auto multiple_sum(Array const& array)
{
  auto rows = array.shape()[0];
  auto cols = array.shape()[1];

  xtensor<double, 2> out = zeros<double>({rows, cols});

  for(decltype(rows) row = 0; row < rows; ++row) {
    view(out, row, all()) = sum(array - view(array, row, all()), /*axis=*/0);
  }
  return out;
}

pytensor<double, 2> py_multiple_sum(pytensor<double, 2> const& array)
{
  return multiple_sum(array);
}

PYBIND11_MODULE(xtensor_multiple_sum, m)
{
    import_numpy();
    m.def("multiple_sum", py_multiple_sum);
}
