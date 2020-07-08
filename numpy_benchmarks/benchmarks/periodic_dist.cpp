#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xpad.hpp"
#include "xtensor/xindex_view.hpp"

#define FORCE_IMPORT_ARRAY

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"


using namespace xt;

template<typename X, typename Y, typename Z>
auto periodic_dist(X const& x, Y const& y, Z const& z, long L, bool periodicX, bool periodicY, bool periodicZ)
{
  auto N = x.size();
  using shape_type = decltype(N);

  xarray<double> xtemp = tile(x, N);
  xtemp.reshape({N, N});
  xtensor<double, 2> dx = xtemp - transpose(xtemp);

  xarray<double> ytemp = tile(y, N);
  ytemp.reshape({N, N});
  xtensor<double, 2> dy = ytemp - transpose(ytemp);

  xarray<double> ztemp = tile(z, N);
  ztemp.reshape({N, N});
  xtensor<double, 2> dz = ztemp - transpose(ztemp);

  if(periodicX) {
    filter(dx, dx>L/2) = filter(dx, dx > L/2) - L;
    filter(dx, dx<-L/2) = filter(dx, dx < -L/2) + L;
  }

  if(periodicY) {
    filter(dy, dy>L/2) = filter(dy, dy > L/2) - L;
    filter(dy, dy<-L/2) = filter(dy, dy < -L/2) + L;
  }

  if(periodicZ) {
    filter(dz, dz>L/2) = filter(dz, dz > L/2) - L;
    filter(dz, dz<-L/2) = filter(dz, dz < -L/2) + L;
  }

  xtensor<double, 2> d = sqrt(square(dx) + square(dy) + square(dz));
  filter(d, equal(d, 0)) = -1;

  return std::make_tuple(d, dx, dy, dz);
}

std::tuple<pytensor<double, 2>, pytensor<double, 2>, pytensor<double, 2>, pytensor<double, 2>>
py_periodic_dist(pytensor<double, 1> const& x, pytensor<double, 1> const& y, pytensor<double, 1> const& z, long L, bool periodicX, bool periodicY, bool periodicZ)
{
  return periodic_dist(x, y, z, L, periodicX, periodicY, periodicZ);
}

PYBIND11_MODULE(xtensor_periodic_dist, m)
{
    import_numpy();
    m.def("periodic_dist", py_periodic_dist);
}

