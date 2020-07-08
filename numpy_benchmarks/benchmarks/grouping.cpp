#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xadapt.hpp"
#define FORCE_IMPORT_ARRAY

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"


using namespace xt;

template<typename Values>
auto grouping(Values const& values)
{
  auto diff = concatenate(xtuple(adapt(std::array<uint32_t, 1>{1}, {1}), xt::diff(values)));
  auto vwdiff0 = where(diff)[0];
  auto wdiff0 = adapt(vwdiff0, {vwdiff0.size()});
  auto idx = eval(concatenate(xtuple(wdiff0, adapt(std::array<uint32_t, 1>{values.size()}, {1}))));
  return std::make_tuple(eval(index_view(values, view(idx, range(xnone(), -1)))), xt::diff(idx));
}

std::tuple<pytensor<uint32_t, 1>, pytensor<uint32_t, 1>> py_grouping(pytensor<uint32_t, 1> const& values)
{
  return grouping(values);
}

PYBIND11_MODULE(xtensor_grouping, m)
{
    import_numpy();
    m.def("grouping", py_grouping);
}
