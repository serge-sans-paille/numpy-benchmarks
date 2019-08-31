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

template<typename Pts>
auto pairwise(Pts const& pts)
{
  return eval(sqrt(sum(square(view(pts, newaxis(), all()) - view(pts, all(), newaxis())), /*axis=*/-1)));
}

pytensor<double, 2> py_pairwise(pytensor<double, 2> const& pts)
{
  return pairwise(pts);
}

PYBIND11_MODULE(xtensor_pairwise, m)
{
    import_numpy();
    m.def("pairwise", py_pairwise);
}
