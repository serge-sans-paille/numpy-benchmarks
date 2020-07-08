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
auto rosen(X const& x)
{
  auto t0 = 100 * square(view(x, range(1, xnone())) - square(view(x, range(xnone(), -1))));
  auto t1 = square(1 - view(x, range(xnone(), -1)));
  return sum(t0 + t1)();

}

double py_rosen(pytensor<double, 1> const& x)
{
  return rosen(x);
}

PYBIND11_MODULE(xtensor_rosen, m)
{
    import_numpy();
    m.def("rosen", py_rosen);
}
