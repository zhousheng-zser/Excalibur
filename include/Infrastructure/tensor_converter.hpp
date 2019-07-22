#pragma once

#include <cassert>
#include <algorithm>

#include <glasssix/tensor.hpp>

namespace glasssix
{
    namespace excalibur
    {
        /// <summary>
        /// Indicate a CPU-based convertion.
        /// </summary>
        struct tensor_cpu_tag{};

        /// <summary>
        /// Indicate a GPU-based convertion.
        /// </summary>
        struct tensor_gpu_tag{};

        /// <summary>
        /// The default converter for underlying types.
        /// </summary>
        template<typename TSource, typename TDestination>
        struct tensor_underlying_type_converter
        {
            TDestination operator()(const TSource& source) const
            {
                return (TDestination)source;
            }
        };
        
        template<typename TSource, typename TDestination>
        tensor_underlying_type_converter<TSource, TDestination> tensor_underlying_type_converter_v;

        /// <summary>
        /// The default converter for tensors.
        /// </summary>
        template<typename TSource, typename TDestination, typename TTag>
        struct tensor_converter
        {
            //static_assert(false, "No appropriate instance of the converter template.");
        };

        /// <summary>
        /// The default CPU-based converter for tensors.
        /// </summary>
        template<typename TSource, typename TDestination>
        struct tensor_converter<TSource, TDestination, tensor_cpu_tag>
        {
            void operator()(const tensor<TSource>& source, tensor<TDestination>& destination) const
            {
                assert(source.device() < 0 && destination.count() < 0);

                auto source_ptr = source.cpu_data();
                auto destination_ptr = destination.mutable_cpu_data();
                auto count = std::min(source.count(), destination.count());

                // Assign the data directly.
                for (size_t i = 0; i < count; i++)
                {
                    destination_ptr[i] = tensor_underlying_type_converter_v<TSource, TDestination>(source_ptr[i]);
                }
            }
        };

        /// <summary>
        /// The default GPI-based converter for tensors.
        /// </summary>
        template<typename TSource, typename TDestination>
        struct tensor_converter<TSource, TDestination, tensor_gpu_tag>
        {
            void operator()(const tensor<TSource>& source, tensor<TDestination>& destination) const
            {
                assert(source.device() >= 0 && destination.count() >= 0);

                auto source_ptr = source.gpu_data();
                auto destination_ptr = destination.mutable_gpu_data();
                auto count = std::min(source.count(), destination.count());

                //...
            }
        };

        template<typename TSource, typename TDestination, typename TTag>
        tensor_converter<TSource, TDestination, TTag> tensor_converter_v;
    }
}
