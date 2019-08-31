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
auto l1norm(X const& x, Y const& y)
{
  return sum(abs(view(x, all(), newaxis(), all()) - y), /*axis=*/-1);
}

pytensor<double, 2> py_l1norm(pytensor<double, 2> const& x, pytensor<double, 2> const& y)
{
  return l1norm(x, y);
}

PYBIND11_MODULE(xtensor_l1norm, m)
{
    import_numpy();
    m.def("l1norm", py_l1norm);
}
