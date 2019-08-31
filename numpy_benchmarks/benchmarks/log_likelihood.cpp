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

template<typename Data>
auto log_likelihood(Data const& data, double mean, double sigma)
{
  auto s = pow((data - mean), 2) / (2 * (sigma * sigma));
  xtensor<double, 1> pdfs = exp(-s);
  pdfs /= std::sqrt(2 * numeric_constants<double>::PI) * sigma;
  pdfs = log(pdfs);
  return sum(pdfs, xt::evaluation_strategy::immediate)();
}

double py_log_likelihood(pytensor<double, 1> const& data, double mean, double sigma)
{
  return log_likelihood(data, mean, sigma);
}

PYBIND11_MODULE(xtensor_log_likelihood, m)
{
    import_numpy();
    m.def("log_likelihood", py_log_likelihood);
}

