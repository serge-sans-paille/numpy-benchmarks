#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor-blas/xlinalg.hpp"

#define FORCE_IMPORT_ARRAY

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

using namespace xt;

template<typename X1, typename X2>
auto euclidean_distance_square(X1 const& x1, X2 const& x2) {
    return -2*linalg::dot(x1, transpose(x2)) + xt::view(sum(square(x1), /*axis=*/1), all(), newaxis()) + sum(square(x2), /*axis=*/1);
}
// FIXME: report the fact that invalid conversion is silent
pytensor<double, 2> py_euclidean_distance_square(pytensor<double, 2>& x1, pytensor<double, 2>& x2)
{
  return euclidean_distance_square(x1, x2);
}

PYBIND11_MODULE(xtensor_euclidean_distance_square, m)
{
    import_numpy();
    m.def("euclidean_distance_square", py_euclidean_distance_square);
}
