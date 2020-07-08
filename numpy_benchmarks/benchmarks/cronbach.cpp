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

template<typename ItemScores>
auto cronbach(ItemScores const& itemscores)
{
  auto itemvars = variance(itemscores, /*axis=*/{1}, /*ddof=*/1);
  auto tscores = sum(itemscores, /*axis=*/{0});
  double nitems = itemscores.shape()[0];
  return nitems / (nitems - 1) * (1 - sum(itemvars) / variance(tscores, /*ddof=*/1))[0];
}

double py_cronbach(pytensor<double, 2> const& itemscores)
{
  return cronbach(itemscores);
}

PYBIND11_MODULE(xtensor_cronbach, m)
{
    import_numpy();
    m.def("cronbach", py_cronbach);
}
