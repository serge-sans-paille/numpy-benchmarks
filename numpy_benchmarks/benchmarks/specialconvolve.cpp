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

template<typename A>
auto specialconvolve(A const& a)
{
  xtensor<uint32_t, 2> rowconvol = view(a, range(1, -1), all()) + view(a, range(xnone(), -2), all()) + view(a, range(2, xnone()), all());
  xtensor<uint32_t, 2> colconvol = view(rowconvol, all(), range(1, -1)) + view(rowconvol, all(), range(xnone(), -2)) + view(rowconvol, all(), range(2, xnone())) - 9 * view(a, range(1, -1), range(1, -1));
  return colconvol;
}

pytensor<uint32_t, 2> py_specialconvolve(pytensor<uint32_t, 2> const& a)
{
  return specialconvolve(a);
}

PYBIND11_MODULE(xtensor_specialconvolve, m)
{
    import_numpy();
    m.def("specialconvolve", py_specialconvolve);
}
