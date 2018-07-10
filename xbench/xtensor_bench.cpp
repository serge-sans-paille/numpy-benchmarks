#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xfixed.hpp"
#define FORCE_IMPORT_ARRAY

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

namespace xt
{
	template <class T, std::size_t N>
	using tensor = xt::pytensor<T, N, xt::layout_type::row_major>;
}

#include "../benchmarks/allpairs_distances_loops.hpp"
#include "../benchmarks/arc_distance.hpp"
#include "../benchmarks/laplacien.hpp"
#include "../benchmarks/diffusion.hpp"
#include "../benchmarks/log_likelihood.hpp"
#include "../benchmarks/reverse_cumsum.hpp"


namespace py = pybind11;

auto xtensor_laplacien(xt::pytensor<double, 3>& arr)
{
    return xt::impl::laplacian_view(arr);
}

auto testfun()
{
	xsimd::batch<double, 4> a(1,2,3,4);
	a = xsimd::sin(a);
	std::cout << a << std::endl;
}

auto cumsum(xt::tensor<double, 1>& a)
{
    return xt::cumsum(a);
}

PYBIND11_MODULE(xbench, m)
{
    xt::import_numpy();
    m.def("laplacien", xtensor_laplacien);
    m.def("diffusion", xt::diffusion);
    m.def("arc_distance", xt::arc_distance);
    m.def("log_likelihood", xt::log_likelihood);
    m.def("reverse_cumsum", xt::reverse_cumsum);
    m.def("cumsum", cumsum);
    m.def("allpairs_distances_loops", xt::allpairs_distances_loops);
}