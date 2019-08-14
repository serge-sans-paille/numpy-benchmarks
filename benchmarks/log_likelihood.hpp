namespace xt
{
	tensor<double, 1> log_likelihood(const tensor<double, 1>& data, double mean, double sigma)
	{
		auto s = pow((data - mean), 2) / (2 * (sigma * sigma));
		tensor<double, 1> pdfs = exp(-s);
#if 0
		pdfs /= std::sqrt(2 * numeric_constants<double>::PI) * sigma;
#else
		pdfs = pdfs / std::sqrt(2 * numeric_constants<double>::PI) * sigma;
#endif
		pdfs = log(pdfs);
		return sum(pdfs, xt::evaluation_strategy::immediate);
	}
}
