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

template<typename Masspoints, typename Dt, typename Plunk, typename Which>
void physics(Masspoints & masspoints, Dt const& dt, Plunk const& plunk, Which const& which)
{
  auto ppos = view(masspoints, 1);
  auto cpos = view(masspoints, 0);
  auto N = cpos.shape()[0];
  auto HOOKE_K = 2100000.;
  auto DAMPING = 0.0001;
  auto MASS = .01;

  xtensor<double, 2> force = zeros<double>({N, (decltype(N))2});
  for(std::size_t i = 1; i < N; ++i)
  {
    auto tmp = view(cpos, i) - view(cpos, i - 1);
    auto dx = tmp(0), dy = tmp(1);
    auto dist = sqrt(square(dx) + square(dy));
    assert(dist != 0);
    auto fmag = -HOOKE_K * dist;
    auto cosine = dx / dist;
    auto sine = dy / dist;
    xtensor<double, 1> fvec = {eval(fmag * cosine)(), eval(fmag * sine)()};
    view(force, i - 1) -= fvec;
    view(force, i) += fvec;
  }
  view(force, 0) = 0;
  view(force, force.shape(0)-1) = xtensor<double, 1>{{0, 0}}; // -1 not supported, see https://github.com/xtensor-stack/xtensor/issues/993
  view(force, which, 1) += plunk;
  auto accel = force / MASS;

  //// verlet integration
  auto npos = (2 - DAMPING) * cpos - (1 - DAMPING) * ppos + accel * (square(dt));

  view(masspoints, 1) = cpos;
  view(masspoints, 0) = npos;
}

template<typename PARTICLE_COUNT_T>
auto wave(PARTICLE_COUNT_T const& PARTICLE_COUNT)
{
    auto SUBDIVISION = 300;
    auto FRAMERATE = 60;
    std::size_t count = PARTICLE_COUNT;
    auto width = 1200;
    auto height = 400;

    xtensor<double, 3> masspoints = empty<double>({(std::size_t)2, count, (std::size_t)2});
    xtensor<double, 1> initpos = zeros<double>({count});
    for(std::size_t i = 1; i < count; ++i)
        initpos[i] = initpos[i - 1] + double(width) / count;
    view(masspoints, all(), all(), 0) = initpos;
    view(masspoints, all(), all(), 1) = height / 2;
    auto f = 15;
    auto plunk_pos = count / 2;
    physics( masspoints, 1./ (SUBDIVISION * FRAMERATE), f, plunk_pos);
    return masspoints;
}

pytensor<double, 3> py_wave(long PARTICLE_COUNT)
{
  return wave(PARTICLE_COUNT);
}

PYBIND11_MODULE(xtensor_wave, m)
{
    import_numpy();
    m.def("wave", py_wave);
}
