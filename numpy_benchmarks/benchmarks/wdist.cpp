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

template<typename AT, typename BT, typename WT>
auto wdist(AT const& A, BT const& B, WT const& W)
{
  auto k = A.shape()[0];
  auto m = A.shape()[1];
  auto n = B.shape()[1];

  xtensor<double, 2> D = zeros<double>({m, n});
  for(decltype(m) ii = 0; ii < m; ++ii) {
    for(decltype(n) jj = 0; jj < n; ++jj) {
      auto wdiff = (view(A, all(), ii) - view(B, all(), jj)) / view(W, all(), ii);
      view(D, ii, jj) = sqrt(sum(square(wdiff)));
    }
  }
  return D;
}

pytensor<double, 2> py_wdist(pytensor<double, 2> const& A, pytensor<double, 2> const& B, pytensor<double, 2> const& C)
{
  return wdist(A, B, C);
}

PYBIND11_MODULE(xtensor_wdist, m)
{
    import_numpy();
    m.def("wdist", py_wdist);
}

