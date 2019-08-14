namespace xt
{
    namespace impl
    {
        template <class T>
        struct lapl_return_type
        {
            using type = T;
        };

        template <class T, std::size_t X, std::size_t Y, std::size_t Z>
        struct lapl_return_type<xt::xtensor_fixed<T, fixed_shape<X, Y, Z>>>
        {
            using type = xtensor_fixed<T, xshape<X - 2, Y - 2, Z>>;
        };

        template <class T>
        using lapl_return_type_t = typename lapl_return_type<T>::type;

        template <class T>
        void print_type(T)
        {
            std::cout << __PRETTY_FUNCTION__ << std::endl;
        }

        template <class T>
        inline lapl_return_type_t<T> laplacian_view(const T& image)
        {
            lapl_return_type_t<T> out_image = xt::abs(4 * view(image, range(1, -1), range(1, -1)) -
                                      view(image, range(0, -2), range(1, -1)) - view(image, range(2, xnone()), range(1, -1)) -
                                      view(image, range(1, -1), range(0, -2)) - view(image, range(1, -1), range(2, xnone()))
                          );
            
            auto valmax = xt::amax(out_image, xt::evaluation_strategy::immediate)();
            valmax = std::max(1., valmax) + 1e-9;
#if 0
            out_image /= valmax;
#else
            out_image = out_image / valmax;
#endif
            return out_image;
        }

        template <class T>
        inline lapl_return_type_t<T> laplacian_strided_view(const T& image)
        {
            lapl_return_type_t<T> out_image = xt::abs(4 * strided_view(image, {range(1, -1), range(1, -1)}) -
                                      strided_view(image, {range(0, -2), range(1, -1)}) - strided_view(image, {range(2, xnone()), range(1, -1)}) -
                                      strided_view(image, {range(1, -1), range(0, -2)}) - strided_view(image, {range(1, -1), range(2, xnone())})
                          );
            
            auto valmax = xt::amax(out_image, xt::evaluation_strategy::immediate)();
            valmax = std::max(1., valmax) + 1e-9;
            out_image /= valmax;
            return out_image;
        }

        template <class T>
        inline lapl_return_type_t<T> laplacian_manual(const T& image)
        {
            using RT = lapl_return_type_t<T>;
            RT out_image = RT({image.shape()[0] - 2, image.shape()[1] - 2, 3}, layout_type::row_major);
            for (std::size_t i = 0; i < out_image.shape()[0]; ++i)
            {
                for (std::size_t j = 0; j < out_image.shape()[1]; ++j)
                {
                    for (std::size_t k = 0; k < 3; ++k)
                    {
                        out_image.unchecked(i, j, k) = std::abs(4 * image.unchecked(i + 1, j + 1, k) - 
                                                                    image.unchecked(i, j + 1, k) - image.unchecked(i + 2, j + 1, k) -
                                                                    image.unchecked(i + 1, j, k) - image.unchecked(i + 1, j + 2, k));
                    }
                }
            }
            auto valmax = xt::amax(out_image, xt::evaluation_strategy::immediate)();
            valmax = std::max(1., valmax) + 1e-9;
            out_image /= valmax;
            return out_image;
        }
    }
}
