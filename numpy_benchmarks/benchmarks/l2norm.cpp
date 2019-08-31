
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
auto l2norm(X const& x)
{
  return sqrt(sum(square(abs(x)), /*axis=*/1));
}

pytensor<double, 1> py_l2norm(pytensor<double, 2> const& x)
{
  return l2norm(x);
}

PYBIND11_MODULE(xtensor_l2norm, m)
{
    import_numpy();
    m.def("l2norm", py_l2norm);
}
