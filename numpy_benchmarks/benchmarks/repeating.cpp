#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xmanipulation.hpp"

#define FORCE_IMPORT_ARRAY

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"


using namespace xt;

template<typename X, typename NVAR_Y>
auto repeating(X const& x, NVAR_Y const& nvar_y)
{
  auto nvar_x = x.shape()[0];
  auto y = empty<double>({nvar_x * (1 + nvar_y)});
  view(y, range(0, nvar_x)) = view(x, range(0, nvar_x));
  // FIXME: unwanted const_cast
  view(y, range(nvar_x, xnone())) = view(repeat(const_cast<X&>(x), nvar_y, /*axis=*/0), all());
  return y;
}

pytensor<double, 1> py_repeating(pytensor<double, 1> const& x, size_t nvar_y)
{
  return repeating(x, nvar_y);
}

PYBIND11_MODULE(xtensor_repeating, m)
{
    import_numpy();
    m.def("repeating", py_repeating);
}


