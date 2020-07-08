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
auto reverse_cumsum(X const& x)
{
  return view(cumsum(view(x, range(xnone(), xnone(), -1))), range(xnone(), xnone(), -1));
}

pytensor<double, 1> py_reverse_cumsum(pytensor<double, 1> const& x)
{
  return reverse_cumsum(x);
}

PYBIND11_MODULE(xtensor_reverse_cumsum, m)
{
    import_numpy();
    m.def("reverse_cumsum", py_reverse_cumsum);
}
