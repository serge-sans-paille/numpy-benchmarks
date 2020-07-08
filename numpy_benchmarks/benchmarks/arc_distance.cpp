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

template<typename T1, typename P1, typename T2, typename P2>
auto arc_distance(T1 const& theta_1, P1 const& phi_1, T2 const& theta_2, P2 const& phi_2)
{
    auto temp = pow(sin((theta_2 - theta_1) / 2), 2) + cos(theta_1) * cos(theta_2) * pow(sin((phi_2 - phi_1) / 2), 2);
    xtensor<double, 1> distance_matrix = 2 * (atan2(sqrt(temp), sqrt(1 - temp)));
    return distance_matrix;

}

pytensor<double, 1> py_arc_distance(pytensor<double, 1> const& theta_1, pytensor<double, 1> const& phi_1, pytensor<double, 1> const& theta_2, pytensor<double, 1> const& phi_2)
{
  return arc_distance(theta_1, phi_1, theta_2, phi_2);
}

PYBIND11_MODULE(xtensor_arc_distance, m)
{
    import_numpy();
    m.def("arc_distance", py_arc_distance);
}


