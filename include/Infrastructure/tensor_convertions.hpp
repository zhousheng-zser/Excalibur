#pragma once

#include "tensor_converter.hpp"
#include "tensor_or_shared.hpp"

#include <memory>

#include <glasssix/tensor.hpp>

namespace glasssix
{
    namespace excalibur
    {
        /// <summary>
        /// Indicate a type which is the destination type of the tensor convertion.
        /// </summary>
        template<typename TDestination>
        struct tensor_convert_to_tag {};

        /// <summary>
        /// Indicate a layout type which a tensor is converted to.
        /// </summary>
        template<tensor_layout layout>
        struct tensor_convert_layout_to_tag {};

        /// <summary>
        /// A template variable to simplify coding.
        /// </summary>
        template<typename TDestination>
        tensor_convert_to_tag<TDestination> tensor_convert_to;

        /// <summary>
        /// A template variable to simplify coding.
        /// </summary>
        template<tensor_layout layout>
        tensor_convert_layout_to_tag<layout> tensor_convert_layout_to;

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
        inline constexpr void assert_numeric()
        {
            static_assert(std::is_arithmetic_v<TArgs>, "All parameters must be numerical.");
        }

        /// <summary>
        /// An internal function to convert the underlying type of a tensor to another within a duplication.
        /// </summary>
        /// <param name="source">The source tensor</param>
        /// <param name="tag">The tag containing information about the destination underlying type</param>
        /// <returns>The destination tensor</returns>
        template<typename TSource, typename TDestination, bool shared>
        auto convert_to_core(const tensor_or_shared<TSource, shared>& source, const tensor_convert_to_tag<TDestination>& tag)
        {
            assert_numeric<TSource>();
            assert_numeric<TDestination>();
            
            auto destination = allocate_tensor<TDestination, shared>(source.access());
            tensor_or_shared<TDestination, shared> destination_wrapper{ destination };
            
            if (source->device() < 0)
            {
                tensor_converter_v<TSource, TDestination, tensor_cpu_tag>(source.access(), destination_wrapper.access());
            }
            else
            {
                tensor_converter_v<TSource, TDestination, tensor_cpu_tag>(source.access(), destination_wrapper.access());
            }

            return destination;
        }

        /// <summary>
        /// An internal function to convert the layout of a tensor to another within a duplication.
        /// </summary>
        /// <param name="source">The source tensor</param>
        /// <param name="tag">The tag containing information about the destination underlying type</param>
        /// <returns>The destination tensor</returns>
        template<bool shared, typename TSource, tensor_layout layout>
        auto convert_layout_to_core(const tensor_or_shared<TSource, shared>& source, const tensor_convert_layout_to_tag<layout>& tag)
        {
            assert_numeric<TSource>();

            auto destination = allocate_tensor<TSource, shared>(source.access());
            tensor_or_shared<TSource, shared> destination_wrapper{ destination };

            if (source->device() < 0)
            {
                tensor_layout_converter_v<TSource, layout, tensor_cpu_tag>(source.access(), destination_wrapper.access());
            }
            else
            {
                tensor_layout_converter_v<TSource, layout, tensor_cpu_tag>(source.access(), destination_wrapper.access());
            }

            return destination;
        }

        /// <summary>
        /// Provide support for convertions of the underlying type for a tensor.
        /// </summary>
        /// <param name="source">The source tensor</param>
        /// <param name="tag">The tag containing information about the destination underlying type</param>
        /// <returns>The destination tensor</returns>
        template<typename TSource, typename TDestination>
        inline tensor<TDestination> operator|(const tensor<TSource>& source, const tensor_convert_to_tag<TDestination>& tag)
        {
            return convert_to_core<TSource, TDestination, false>(source, tag);
        }

        /// <summary>
        /// Provide support for convertions of the underlying type for a tensor.
        /// </summary>
        /// <param name="source">The source tensor</param>
        /// <param name="tag">The tag containing information about the destination underlying type</param>
        /// <returns>The destination tensor</returns>
        template<typename TSource, typename TDestination>
        inline std::shared_ptr<tensor<TDestination>> operator|(const std::shared_ptr<tensor<TSource>>& source, const tensor_convert_to_tag<TDestination>& tag)
        {
            return convert_to_core<TSource, TDestination, true>(source, tag);
        }

        /// <summary>
        /// Provide support for convertions of the layout for a tensor.
        /// </summary>
        /// <param name="source">The source tensor</param>
        /// <param name="tag">The tag containing information about the destination layout</param>
        /// <returns>The destination tensor</returns>
        template<typename TSource, tensor_layout layout>
        inline tensor<TSource> operator|(const tensor<TSource>& source, const tensor_convert_layout_to_tag<layout>& tag)
        {
            return convert_layout_to_core<TSource, layout, false>(source, tag);
        }

        /// <summary>
        /// Provide support for convertions of the layout for a tensor.
        /// </summary>
        /// <param name="source">The source tensor</param>
        /// <param name="tag">The tag containing information about the destination layout</param>
        /// <returns>The destination tensor</returns>
        template<typename TSource, tensor_layout layout>
        inline std::shared_ptr<tensor<TSource>> operator|(const std::shared_ptr<tensor<TSource>>& source, const tensor_convert_layout_to_tag<layout>& tag)
        {
            return convert_layout_to_core<TSource, layout, true>(source, tag);
        }
    }
}
