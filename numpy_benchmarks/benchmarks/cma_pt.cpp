#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xfixed.hpp"
#define FORCE_IMPORT_ARRAY

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

#include <cassert>


using namespace xt;

template<typename Et, typename Wx>
auto apply_filter_pt(Et const& E, Wx const& wx) {
  auto Xest = sum(E * conj(wx))();
  return Xest;
}

template<typename Et, typename Wxy, typename Mu, typename Rt, typename Os>
auto cma_pt(Et const& E, Wxy & wxy, Mu const& mu, Rt const& R, Os const& os)
{
    std::size_t pols = E.shape()[0];
    std::size_t L = E.shape()[1];
    assert(wxy.shape()[0] == pols);
    assert(wxy.shape()[1] == pols);
    std::size_t ntaps = wxy.shape()[2];
    std::size_t N = (L/os/ntaps-1)*ntaps;
    xtensor<std::complex<double>, 2> err = zeros<std::complex<double>>({pols, L});
    for(std::size_t k = 0; k < pols; ++k) {
      for(std::size_t i = 0; i < N; ++i) {
            auto X = view(E, all(), range(i*os,i*os+ntaps));
            auto Xest = apply_filter_pt(X, wxy[k]);
            err(k,i) = (R-square(abs(Xest))())*Xest;
            view(wxy, k) += mu*conj(err(k,i))*X;
      }
    }
    return std::make_tuple(err, wxy);
}

std::tuple<pytensor<std::complex<double>, 2>, pytensor<std::complex<double>, 3>>
py_cma_pt(
    pytensor<std::complex<double>, 2> const& E,
    pytensor<std::complex<double>, 3> & wxy,
    double mu,
    double R,
    long os)
{
  return cma_pt(E, wxy, mu, R, os);
}

PYBIND11_MODULE(xtensor_cma_pt, m)
{
    import_numpy();
    m.def("cma_pt", py_cma_pt);
}
