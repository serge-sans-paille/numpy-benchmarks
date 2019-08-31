#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xadapt.hpp"

#define FORCE_IMPORT_ARRAY

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"


using namespace xt;

template<typename DB, typename Mask>
auto check_mask(DB const& db, Mask const& mask)
{
  xtensor<bool, 1> out = zeros<bool>({db.shape()[0]});
  for(std::size_t idx = 0; idx < out.size(); ++idx) {
    auto target = db(idx, 0);
    auto vector = view(db, idx, range(1, all()));
    if(all(equal(mask, vector & mask)))
      if(target == 1)
        out[idx] = 1;
  }
  return out;
}

pytensor<bool, 1> py_check_mask(pytensor<bool, 2> const& db)
{
  std::array<bool, 3> mask = {1, 0, 1};
  return check_mask(db, xt::adapt(mask, {3}));
}

PYBIND11_MODULE(xtensor_check_mask, m)
{
    import_numpy();
    m.def("check_mask", py_check_mask);
}

