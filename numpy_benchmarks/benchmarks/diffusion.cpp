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

template<typename U, typename TempU>
void diffusion(U& u, TempU& tempU, int iterNum)
{
  float mu = 0.1;
  for (int n = 0; n < iterNum; ++n) {
    view(tempU, range(1, -1), range(1, -1)) =
        view(u, range(1, -1), range(1, -1)) +
        mu * (view(u, range(2, xnone()), range(1, -1)) -
              2 * view(u, range(1, -1), range(1, -1)) +
              view(u, range(0, -2), range(1, -1)) +
              view(u, range(1, -1), range(2, xnone())) -
              2 * view(u, range(1, -1), range(1, -1)) +
              view(u, range(1, -1), range(0, -2)));
    u = tempU;
    tempU.fill(0.f);
  }
}

void py_diffusion(pytensor<float, 2>& u, pytensor<float, 2>& tempU, int iterNum)
{
  return diffusion(u, tempU, iterNum);
}

PYBIND11_MODULE(xtensor_diffusion, m)
{
    import_numpy();
    m.def("diffusion", py_diffusion);
}


