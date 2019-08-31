#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xsort.hpp"
#define FORCE_IMPORT_ARRAY

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"


using namespace xt;

template<typename ET, typename Symbols>
auto make_decision(ET const& E, Symbols const& symbols)
{
  auto L = E.shape()[0];
  xtensor<typename ET::value_type, 1> syms_out = zeros<typename ET::value_type>({L});
  for(decltype(L) i = 0; i < L; ++i) {
    auto im = argmin(square(abs(E[i] - symbols)));
    syms_out[i] = symbols[im];
  }
  return syms_out;
}

pytensor<std::complex<double>, 1>
py_make_decision(pytensor<std::complex<double>, 1> const& E, pytensor<std::complex<double>, 1> const& symbols)
{
  return make_decision(E, symbols);
}

PYBIND11_MODULE(xtensor_make_decision, m)
{
    import_numpy();
    m.def("make_decision", py_make_decision);
}
