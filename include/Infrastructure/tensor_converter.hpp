#pragma once

#include <cassert>
#include <algorithm>

#include <glasssix/tensor.hpp>

#include "tensor_helper.hpp"
#include "tensor_layout.hpp"

namespace glasssix
{
    namespace excalibur
    {
        /// <summary>
        /// Indicate a CPU-based convertion.
        /// </summary>
        struct tensor_cpu_tag {};

        /// <summary>
        /// Indicate a GPU-based convertion.
        /// </summary>
        struct tensor_gpu_tag {};

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
        /// The default converter.
        /// </summary>
        template<typename TSource, typename TDestination, typename TTag>
        struct tensor_converter {};

        /// <summary>
        /// The default CPU-based converter.
        /// </summary>
        template<typename TSource, typename TDestination>
        struct tensor_converter<TSource, TDestination, tensor_cpu_tag>
        {
            void operator()(const tensor<TSource>& source, tensor<TDestination>& destination) const
            {
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

        template<typename TSource, typename TDestination, typename TTag>
        tensor_converter<TSource, TDestination, TTag> tensor_converter_v;

        /// <summary>
        /// The default layout converter.
        /// </summary>
        template<typename TUnderlyingType, tensor_layout layout, typename TTag>
        struct tensor_layout_converter {};

        /// <summary>
        /// The default CPU-based layout converter.
        /// Layout type: grayscale 8-bit.
        /// </summary>
        template<typename TUnderlyingType>
        struct tensor_layout_converter<TUnderlyingType, tensor_layout::grayscale, tensor_cpu_tag>
        {
            void operator()(const tensor<TUnderlyingType>& source, tensor<TUnderlyingType>& destination) const
            {
                // We only support triple-channel bitmaps and single-channel bitmaps.
                assert(tensor_helper::has_single_channel(source) || tensor_helper::has_triple_channel(source) || tensor_helper::has_quadruple_channel(source));

                switch (source.channels())
                {
                case 1:
                    destination = std::move(source.clone());
                    break;
                case 3:
                    tensor_helper::rgb_or_rgba_to_gray(source, destination, 1);
                    break;
                case 4:
                    tensor_helper::rgb_or_rgba_to_gray(source, destination, 1);
                    break;
                default:
                    break;
                }
            }
        };

        /// <summary>
        /// The default CPU-based layout converter.
        /// Layout type: grayscale 32-bit.
        /// </summary>
        template<typename TUnderlyingType>
        struct tensor_layout_converter<TUnderlyingType, tensor_layout::grayscale_3, tensor_cpu_tag>
        {
            void operator()(const tensor<TUnderlyingType>& source, tensor<TUnderlyingType>& destination) const
            {
                // We only support triple-channel bitmaps and single-channel bitmaps.
                assert(tensor_helper::has_single_channel(source) || tensor_helper::has_triple_channel(source) || tensor_helper::has_quadruple_channel(source));

                switch (source.channels())
                {
                case 1:
                    destination = std::move(source.clone());
                    break;
                case 3:
                    tensor_helper::rgb_or_rgba_to_gray(source, destination, 3);
                    break;
                case 4:
                    tensor_helper::rgb_or_rgba_to_gray(source, destination, 3);
                    break;
                default:
                    break;
                }
            }
        };

        /// <summary>
        /// The default CPU-based layout converter.
        /// Layout type: RGB.
        /// </summary>
        template<typename TUnderlyingType>
        struct tensor_layout_converter<TUnderlyingType, tensor_layout::rgb, tensor_cpu_tag>
        {
            void operator()(const tensor<TUnderlyingType>& source, tensor<TUnderlyingType>& destination) const
            {
                // We only support triple-channel bitmaps and single-channel bitmaps.
                assert(tensor_helper::has_triple_channel(source) || tensor_helper::has_quadruple_channel(source));

                switch (source.channels())
                {
                case 3:
                    destination = std::move(source.clone());
                    break;
                case 4:
                    tensor_helper::rgba_to_rgb(source, destination);
                    break;
                default:
                    break;
                }
            }
        };

        template<typename TUnderlyingType, tensor_layout layout, typename TTag>
        tensor_layout_converter<TUnderlyingType, layout, TTag> tensor_layout_converter_v;
    }
}
