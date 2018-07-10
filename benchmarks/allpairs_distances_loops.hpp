namespace xt
{
	auto allpairs_distances_loops(const tensor<double, 2>& X, const tensor<double, 2>& Y)
	{
		tensor<double, 2> result;
		result.resize({X.shape()[0], Y.shape()[0]});
		for (size_t i = 0; i < X.shape()[0]; ++i)
			for (size_t j = 0; j < Y.shape()[0]; ++j)
			{
				auto tmp = eval(view(X, i, all()) - view(Y, j, all()));
			    result(i, j) = sum(tmp * tmp)();
			}
		return result;
	}
}