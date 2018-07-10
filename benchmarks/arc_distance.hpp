// #setup: N = 10000 ; import numpy as np ; t0, p0, t1, p1 = np.random.randn(N), np.random.randn(N), np.random.randn(N), np.random.randn(N)
// #run: arc_distance(t0, p0, t1, p1)

// #pythran export arc_distance(float64 [], float64[], float64[], float64[])

// import numpy as np
namespace xt
{
	auto arc_distance(const tensor<double, 1>& theta_1, const tensor<double, 1>& phi_1,
	                  const tensor<double, 1>& theta_2, const tensor<double, 1>& phi_2)
	{
	    // """
	    // Calculates the pairwise arc distance between all points in vector a and b.
	    // """
	    tensor<double, 1> temp = pow(sin((theta_2 - theta_1) / 2), 2) + cos(theta_1) * cos(theta_2) * pow(sin((phi_2 - phi_1) / 2), 2);
	    tensor<double, 1> distance_matrix = 2 * (atan2(sqrt(temp), sqrt(1 - temp)));
	    return distance_matrix;
	}
}