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

template<typename Ru, typename Rv>
auto grayscott(long counts, double Du, double Dv, double F, double k, Ru const& ru, Rv const& rv)
{
  std::size_t n = 280;
  xtensor<float, 2> U = zeros<float>({n+2, n+2});
  xtensor<float, 2> V = zeros<float>({n+2, n+2});
  auto u = view(U, range(1, -1), range(1, -1));
  auto v = view(V, range(1, -1), range(1, -1));

  std::size_t r = 20;
  view(u, all()) = 1.0;
  view(U, range(n / 2 - r, n / 2 + r), range(n/2 - r, r/2 + r)) = 0.50;
  view(V, range(n / 2 - r, n / 2 + r), range(n/2 - r, r/2 + r)) = 0.25;
  u += 0.15*ru;
  v += 0.15*rv;

  for(long i = 0; i < counts; ++i) {
        auto Lu = view(U, range(0, -2), range(1,-1))
                + view(U, range(1, -1), range(0,-2))
                - 4*view(U, range(1,-1),range(1,-1))
                + view(U, range(1, -1), range(2, xnone()))
                + view(U, range(2, xnone()), range(1, -1));
        auto Lv = view(V, range(0, -2), range(1, -1))
                + view(V, range(1, -1), range(0, -2))
                - 4*view(V, range(1, -1), range(1, -1))
                + view(V, range(1, -1), range(2, xnone()))
                + view(V, range(2, all()), range(1, -1));
        auto uvv = eval(u*v*v);
        u += Du*Lu - uvv + F*(1 - u);
        v += Dv*Lv + uvv - (F + k)*v;
  }

  return V;
}

pytensor<float, 2> py_grayscott(long counts, double Du, double Dv, double F, double k, pytensor<double, 2> const& ru, pytensor<double, 2> const& rv)
{
  return grayscott(counts, Du, Dv, F, k, ru, rv);
}

PYBIND11_MODULE(xtensor_grayscott, m)
{
    import_numpy();
    m.def("grayscott", py_grayscott);
}
