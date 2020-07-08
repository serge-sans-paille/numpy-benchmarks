#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xcomplex.hpp"
#define FORCE_IMPORT_ARRAY

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"


using namespace xt;

template<typename A>
auto normalize_complex_arr(A const & a)
{
  auto a_oo = a - amin(real(a))() - std::complex<double>{0., 1.} * amin(imag(a))();
  return eval(a_oo / amax(abs(a_oo))());
}

pytensor<std::complex<double>, 1> py_normalize_complex_arr(pytensor<std::complex<double>, 1> const& a)
{
  return normalize_complex_arr(a);
}

PYBIND11_MODULE(xtensor_normalize_complex_arr, m)
{
    import_numpy();
    m.def("normalize_complex_arr", py_normalize_complex_arr);
}
