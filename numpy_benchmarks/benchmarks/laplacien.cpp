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

template<typename Image>
auto laplacien(Image const& image)
{
  xtensor<double, 3> out_image = abs(4 * view(image, range(1, -1), range(1, -1)) -
                       view(image, range(0, -2), range(1, -1)) -
                       view(image, range(2, xnone()), range(1, -1)) -
                       view(image, range(1, -1), range(0, -2)) -
                       view(image, range(1, -1), range(2, xnone())));
  auto valmax_tmp = amax(out_image, evaluation_strategy::immediate)(); // see https://github.com/QuantStack/xtensor/issues/1744
  auto valmax = std::max(1.,valmax_tmp)+1.e-9;
  out_image /= valmax;
  return out_image;
}

pytensor<double, 3> py_laplacien(pytensor<double, 3> const& image)
{
  return laplacien(image);
}

PYBIND11_MODULE(xtensor_laplacien, m)
{
    import_numpy();
    m.def("laplacien", py_laplacien);
}


