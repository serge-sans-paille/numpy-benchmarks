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

template<typename Grid>
auto laplacian(Grid const& grid)
{
  return roll(grid, +1, 0) + roll(grid, -1, 0) + roll(grid, +1, 1) + roll(grid, -1, 1) - 4 * grid;
}

template<typename Grid, typename Dt>
auto evolve(Grid const& grid, Dt const& dt, long D=1)
{
  return grid + dt * D * laplacian(grid);
}

pytensor<double, 2> py_evolve(pytensor<double, 2> const& grid, double dt)
{
  return evolve(grid, dt);
}

PYBIND11_MODULE(xtensor_evolve, m)
{
    import_numpy();
    m.def("evolve", py_evolve);
}
