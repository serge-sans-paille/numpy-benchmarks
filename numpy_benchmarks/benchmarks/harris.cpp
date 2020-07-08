#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xfixed.hpp"
#define FORCE_IMPORT_ARRAY

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"


using namespace xt;

template<typename IT>
auto harris(IT const& I)
{
  std::size_t m = I.shape()[0], n = I.shape()[1];
  auto dx = view(view(I, range(1, xnone()), all()) - view(I, range(xnone(), m - 1), all()), all(), range(1, xnone()));
  auto dy = view(view(I, all(), range(1, xnone())) - view(I, all(), range(xnone(), n - 1)), range(1, xnone()), all());

  //
  //   At each point we build a matrix
  //   of derivative products
  //   M =
  //   | A = dx^2     C = dx * dy |
  //   | C = dy * dx  B = dy * dy |
  //
  //   and the score at that point is:
  //      det(M) - k*trace(M)^2
  //

  auto A = dx * dx;
  auto B = dy * dy;
  auto C = dx * dy;
  auto tr = A + B;
  auto det = A * B - C * C;
  auto k = 0.05;
  return eval(det - k * tr * tr);
}

pytensor<double, 2>
py_harris(pytensor<double, 2> const& I)
{
  return harris(I);
}

PYBIND11_MODULE(xtensor_harris, m)
{
    import_numpy();
    m.def("harris", py_harris);
}
