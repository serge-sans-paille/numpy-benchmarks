namespace xt
{
    auto diffusion(tensor<float, 2>& u, tensor<float, 2>& tempU, int iterNum)
    {
        // """
        // Apply Numpy matrix for the Forward-Euler Approximation
        // """

        float mu = 0.1;
        for (int n = 0; n < iterNum; ++n)
        {
            view(tempU, range(1, -1), range(1, -1)) = view(u, range(1, -1), range(1, -1)) + mu * 
                                                      (
                                                            view(u, range(2, xnone()), range(1, -1)) - 2 * view(u, range(1, -1), range(1, -1)) + view(u, range(0, -2), range(1, -1)) + 
                                                            view(u, range(1, -1), range(2, xnone())) - 2 * view(u, range(1, -1), range(1, -1)) + view(u, range(1, -1), range(0, -2))
                                                      );
            u = tempU;
            tempU.fill(0.f);
        }
    }
}