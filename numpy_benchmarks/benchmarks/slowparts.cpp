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

template<typename D, typename Re, typename PreDz, typename PreWz, typename SRWT, typename RSWT, typename YxV, typename XyU, typename Resid>
auto slowparts(D const& d, Re const& re, PreDz const& preDz, PreWz const& preWz, SRWT const& SRW, RSWT const& RSW, YxV const& yxV, XyU const& xyU, Resid const& resid)
{
  auto fprime = [](auto x) { return 1 - pow(tanh(x), 2); };

  xtensor<double, 4> partialDU = zeros<double>({d + 1, re, 2 * d, d});
  for(D k = 0; k < 2 * d; ++k) {
    for(D i = 0; i < d; ++i) {
      view(partialDU, all(), all(), k, i) = fprime(preDz[k]) * fprime(preWz[i]) * (SRW(i, k) + RSW(i, k)) * view(yxV, all(), all(), i);
    }
  }
  return partialDU;
}

pytensor<double, 4> py_slowparts(long d, long re, pytensor<double, 3> const& preDz, pytensor<double, 3> const& preWz, pytensor<double, 2> const& SRW, pytensor<double, 2> const& RSW, pytensor<double, 3> const&yxV, pytensor<double, 3> const& xyU, long resid)
{
  return slowparts(d, re, preDz, preWz, SRW, RSW, yxV, xyU, resid);
}

PYBIND11_MODULE(xtensor_slowparts, m)
{
    import_numpy();
    m.def("slowparts", py_slowparts);
}

