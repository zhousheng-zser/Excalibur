#pragma once

#include "tensor_converter.hpp"

#include <memory>

#include <glasssix/tensor.hpp>

namespace glasssix
{
    namespace excalibur
    {
        /// <summary>
        /// Indicate a type which is the destination type of tensor convertions.
        /// </summary>
        template<typename TDestination>
        struct tensor_convert_to_tag {};

        /// <summary>
        /// A template variable to simplify coding.
        /// </summary>
        template<typename TDestination>
        tensor_convert_to_tag<TDestination> tensor_convert_to;

        /// <summary>
        /// Allocate a tensor in a uniform form.
        /// </summary>
        /// <param name="source">The source tensor</param>
        /// <returns>The allocated tensor</returns>
        template<typename TUnderlyingType, bool shared = false>
        auto allocate_tensor(const tensor_& source)
        {
            auto input_vector = source.order() == NHWC ?
                std::vector<int>{ source.num(), source.height(), source.width(), source.channels() } :
                std::vector<int>{ source.num(),  source.channels(),  source.height(),  source.width() };

            if constexpr (shared)
            {
                return std::make_shared<tensor<TUnderlyingType>>(input_vector, source.device(), source.order());
            }
            else
            {
                return tensor<TUnderlyingType>{ input_vector, source.device(), source.order() };
            }
        }

        /// <summary>
        /// Assert the template argumant is numeric statically.
        /// </summary>
        template<typename TArgs>
        constexpr void assert_numeric()
        {
            static_assert(std::is_floating_point_v<TArgs> || std::is_integral_v<TArgs>, "All parameters must be numeric.");
        }

        /// <summary>
        /// Provide support for commonly used convertions between tensors.
        /// </summary>
        /// <param name="source">The source tensor</param>
        /// <param name="tag">The tag containing information about the destination underlying type</param>
        /// <returns>The destination tensor</returns>
        template<typename TSource, typename TDestination>
        tensor<TDestination> operator|(const tensor<TSource>& source, const tensor_convert_to_tag<TDestination>& tag)
        {
            assert_numeric<TSource>();
            assert_numeric<TDestination>();

            auto destination = allocate_tensor<TDestination>(source);

            if (source.device() < 0)
            {
                tensor_converter_v<TSource, TDestination, tensor_cpu_tag>(source, destination);
            }
            else
            {
                tensor_converter_v<TSource, TDestination, tensor_gpu_tag>(source, destination);
            }

            return destination;
        }

        /// <summary>
        /// Provide support for commonly used convertions between tensors.
        /// </summary>
        /// <param name="source">The source tensor</param>
        /// <param name="tag">The tag containing information about the destination underlying type</param>
        /// <returns>The destination tensor</returns>
        template<typename TSource, typename TDestination>
        std::shared_ptr<tensor<TDestination>> operator|(const std::shared_ptr<tensor<TSource>>& source, const tensor_convert_to_tag<TDestination>& tag)
        {
            assert_numeric<TSource>();
            assert_numeric<TDestination>();

            auto destination = allocate_tensor<TDestination, true>(*source);

            if (source->device() < 0)
            {
                tensor_converter_v<TSource, TDestination, tensor_cpu_tag>(*source, *destination);
            }
            else
            {
                tensor_converter_v<TSource, TDestination, tensor_gpu_tag>(*source, *destination);
            }

            return destination;
        }
    }
}
