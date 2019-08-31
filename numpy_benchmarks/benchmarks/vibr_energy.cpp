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

template<typename H, typename A, typename I>
auto vibr_energy(H const& harmonic, A const& anharmonic, I const& i)
{
    return exp(- harmonic * i - anharmonic * ( i * i ) );
}

pytensor<double, 1> py_vibr_energy(pytensor<double, 1> const& harmonic, pytensor<double, 1> const& anharmonic, pytensor<double, 1> const& i)
{
  return vibr_energy(harmonic, anharmonic, i);
}

PYBIND11_MODULE(xtensor_vibr_energy, m)
{
    import_numpy();
    m.def("vibr_energy", py_vibr_energy);
}
