#pragma once

#include "tensor_or_shared.hpp"

#include <cassert>
#include <functional>

#include <glasssix/tensor.hpp>

namespace glasssix
{
    namespace excalibur
    {
        /// <summary>
        /// Some helper functions for building a tensor.
        /// </summary>
        class tensor_helper final
        {
        public:
            /// <summary>
            /// Determine whether the tensor has a triple channel.
            /// </summary>
            /// <param name="tensor">The tensor</param>
            /// <returns>True or false</returns>
            inline static bool has_triple_channel(const tensor_& tensor)
            {
                return tensor.channels() == 3;
            }

            /// <summary>
            /// Determine whether the tensor has a single channel.
            /// </summary>
            /// <param name="tensor">The tensor</param>
            /// <returns>True or false</returns>
            inline static bool has_single_channel(const tensor_& tensor)
            {
                return tensor.channels() == 1;
            }

            /// <summary>
            /// Determine whether the tensor has a quadruple channel.
            /// </summary>
            /// <param name="tensor">The tensor</param>
            /// <returns>True or false</returns>
            inline static bool has_quadruple_channel(const tensor_& tensor)
            {
                return tensor.channels() == 4;
            }

            /// <summary>
            /// Create a tensor base on the specified type and the bitmap data accordingly.
            /// </summary>
            /// <typeparam name="TUnderlyingType">The underlying type</typeparam>
            /// <param name="shared">A value that indicates the result is either an "object" or a "shared_ptr"</param>
            /// <param name="bitmap">The bitmap data</param>
            /// <param name="order">The memory order</param>
            /// <param name="device">The device ID</param>
            /// <param name="width">The width in pixels</param>
            /// <param name="height">The height in pixels</param>
            /// <param name="stride">The number of bytes across a line</param>
            /// <param name="channels">The channel count</param>
            /// <returns>The result</returns>
            template<typename TUnderlyingType, bool shared>
            static auto create(const void* bitmap, orderType order, int device, int width, int height, int stride, int channels)
            {
                static_assert(std::is_arithmetic_v<TUnderlyingType>, "The underlying type of a tensor must be integral or floating-point.");
                assert(bitmap != nullptr);

                // Initialize the tensor parameters.
                auto input_data = reinterpret_cast<const TUnderlyingType*>(bitmap);
                auto input_vector = order == NHWC ? std::vector<int>{ 1, height, width, channels } : std::vector<int>{ 1, channels, height, width };

                // Create a tensor according to the memory order.
                // It may be either an "object" or a "shared_ptr".
                auto result = make_tensor_core<TUnderlyingType, shared>(input_vector, device, order);
                tensor_or_shared<TUnderlyingType, shared> proxy{ result };

                // Copy the data to the tensor.
                auto output_data = proxy.access().mutable_cpu_data();
                copy_data_core<false>(order, input_data, output_data, width, height, channels, stride);

                return result;
            }

            /// <summary>
            /// Copy the data of a tensor to an standard bitmap with an identical buffer.
            /// </summary>
            /// <typeparam name="TUnderlyingType">The underlying type</typeparam>
            /// <param name="data">The tensor</param>
            /// <param name="bitmap">The bitmap buffer</param>
            /// <param name="stride">The stride</param>
            template<typename TUnderlyingType>
            static void copy_to_bitmap(const tensor<TUnderlyingType>& data, void* bitmap, int stride)
            {
                static_assert(std::is_arithmetic_v<TUnderlyingType>, "The underlying type of a tensor must be integral or floating-point.");
                assert(bitmap != nullptr);

                // Initialize necessary buffers.
                auto input_data = data.cpu_data();
                auto output_data = reinterpret_cast<TUnderlyingType*>(bitmap);

                // Copy the data to the bitmap.
                copy_data_core<true>(data.order(), input_data, output_data, data.width(), data.height(), data.channels(), stride);
            }

            /// <summary>
            /// Convert a RGB or a RGBA tensor to a grayscale tensor.
            /// </summary>
            /// <typeparam name="TUnderlyingType">The underlying type</typeparam>
            /// <param name="source">The source tensor</param>
            /// <param name="destination">The destination tensor</param>
            /// <param name="channels">The number of channels in the destination</param>
            template<typename TUnderlyingType>
            static void rgb_or_rgba_to_gray(const tensor<TUnderlyingType>& source, tensor<TUnderlyingType>& destination, int channels)
            {
                static_assert(std::is_arithmetic_v<TUnderlyingType>, "The underlying type of a tensor must be integral or floating-point.");
                assert(has_triple_channel(source) || has_quadruple_channel(source));

                // Initialize necessary constants.
                auto width = source.width();
                auto height = source.height();
                auto source_channels = source.channels();
                auto input_data = source.mutable_cpu_data();

                // Transform the data.
                transform_tensor_core<true>(source, destination, channels, [&](orderType order)
                {
                    return order == NCHW ?
                        std::function{ [&](int w, int h) { return static_cast<TUnderlyingType>(input_data[width * h + w] * 0.299 + input_data[width * height + width * h + w] * 0.587 + input_data[width * height * 2 + width * h + w] * 0.114); } } :
                        std::function{ [&](int w, int h) { return static_cast<TUnderlyingType>(input_data[(width * h + w) * source_channels] * 0.299 + input_data[(width * h + w) * source_channels + 1] * 0.587 + input_data[(width * h + w) * source_channels + 2] * 0.114); } };
                });
            }

            /// <summary>
            /// Convert a RGBA tensor to a RGB tensor.
            /// </summary>
            /// <typeparam name="TUnderlyingType">The underlying type</typeparam>
            /// <param name="source">The source tensor</param>
            /// <param name="destination">The destination tensor</param>
            template<typename TUnderlyingType>
            static void rgba_to_rgb(const tensor<TUnderlyingType>& source, tensor<TUnderlyingType>& destination)
            {
                static_assert(std::is_arithmetic_v<TUnderlyingType>, "The underlying type of a tensor must be integral or floating-point.");
                assert(has_quadruple_channel(source));

                static constexpr int channels = 3;

                // Initialize necessary constants.
                auto width = source.width();
                auto height = source.height();
                auto source_channels = source.channels();
                auto input_data = source.mutable_cpu_data();

                // Transform the data.
                transform_tensor_core<false>(source, destination, channels, [&](orderType order)
                {
                    return order == NCHW ?
                        std::function{ [&](int w, int h, int c) { return input_data[width * height * c + width * h + w]; } } :
                        std::function{ [&](int w, int h, int c) { return input_data[(width * h + w) * source_channels + c]; } };
                });
            }
        private:
            template<bool to_bitmap, typename TUnderlyingType>
            static void copy_data_core(orderType order, const TUnderlyingType* input_data, TUnderlyingType* output_data, int width, int height, int channels, int stride)
            {
                auto channel_bytes = sizeof(TUnderlyingType);
                auto pixel_bytes = channel_bytes * channels;
                auto line_bytes = width * pixel_bytes;
                auto padding_bytes = stride - width * pixel_bytes;
                auto raw_output_data = reinterpret_cast<uint8_t*>(output_data);
                auto raw_input_data = reinterpret_cast<const uint8_t*>(input_data);

                switch (order)
                {
                case NCHW:
                {
                    // For bitmaps, the order is B, G, R
                    for (auto c = channels - 1; c >= 0; c--)
                    {
                        for (auto h = 0; h < height; h++)
                        {
                            for (auto w = 0; w < width; w++)
                            {
                                // In the "to-bitmap" mode, padding bytes are required.
                                if constexpr (to_bitmap)
                                {
                                    *reinterpret_cast<TUnderlyingType*>(raw_output_data + (channels * width * h + channels * w + c) * channel_bytes + padding_bytes * h) = input_data[width * height * c + width * h + w];
                                }
                                else
                                {
                                    output_data[width * height * c + width * h + w] = *reinterpret_cast<const TUnderlyingType*>(raw_input_data + (channels * width * h + channels * w + c) * channel_bytes + padding_bytes * h);
                                }
                            }
                        }
                    }
                    break;
                }
                case NHWC:
                {
                    // Simply do progressive scanning.
                    for (auto h = 0; h < height; h++)
                    {
                        auto index = h * line_bytes;

                        // In the "to-bitmap" mode, padding bytes are required.
                        if constexpr (to_bitmap)
                        {
                            memcpy(raw_output_data + index + padding_bytes * h, raw_input_data + index, line_bytes);
                        }
                        else
                        {
                            memcpy(raw_output_data + index, raw_input_data + index + padding_bytes * h, line_bytes);
                        }
                    }
                    break;
                }
                default:
                    break;
                }
            }

            template<bool merging_pixels, typename TUnderlyingType, typename TPixelGeneratorSelector>
            static void transform_tensor_core(const tensor<TUnderlyingType>& source, tensor<TUnderlyingType>& destination, int channels, TPixelGeneratorSelector&& pixel_generator_selector)
            {
                static_assert(std::is_invocable_v<TPixelGeneratorSelector, orderType>, R"(The selector must contain only one argument with the type of "orderType".)");
                static_assert(std::is_arithmetic_v<TUnderlyingType>, "The underlying type of a tensor must be integral or floating-point.");

                // Check the pixel generator.
                using pixel_generator_type = decltype(pixel_generator_selector(orderType{}));
                if constexpr (merging_pixels)
                {
                    static_assert(std::is_invocable_v<pixel_generator_type, int, int>, R"(In the merging-pixel mode, the "pixel_generator_type" must be a callable type with (int w, int h) as arguments.)");
                }
                else
                {
                    static_assert(std::is_invocable_v<pixel_generator_type, int, int, int>, R"(In the ordinary mode, the "pixel_generator_type" must be a callable type with (int w, int h, int c) as arguments.)");
                }

                assert(has_triple_channel(source) || has_quadruple_channel(source));

                // Initialize necessary constants.
                auto order = source.order();
                auto width = source.width();
                auto height = source.height();
                auto input_data = source.mutable_cpu_data();
                auto input_vector = order == NHWC ? std::vector<int>{ 1, height, width, channels } : std::vector<int>{ 1, channels, height, width };

                // Create a tensor according to the memory order.
                destination = tensor<TUnderlyingType>{ input_vector, source.device(), order };

                // Get the appropriate handlers.
                TUnderlyingType pixel;
                auto pixel_setter = pixel_setter_core(destination);
                auto pixel_generator = std::forward<TPixelGeneratorSelector>(pixel_generator_selector)(order);

                // Copy the data to the tensor.
                for (auto h = 0; h < height; h++)
                {
                    for (auto w = 0; w < width; w++)
                    {
                        // In the merging-pixel mode, we call the pixel generator outside the channel loop.
                        if constexpr (merging_pixels)
                        {
                            pixel = pixel_generator(w, h);
                        }
                        for (auto c = 0; c < channels; c++)
                        {
                            // In the ordinary mode, we call the pixel generator inside the channel loop.
                            // Besides, the handler function contains three arguments with a channel at the tail.
                            if constexpr (!merging_pixels)
                            {
                                pixel = pixel_generator(w, h, c);
                            }
                            pixel_setter(w, h, c, pixel);
                        }
                    }
                }
            }

            template<typename TUnderlyingType>
            inline static auto pixel_setter_core(const tensor<TUnderlyingType>& data)
            {
                return data.order() == NCHW ?
                    std::function{ [&, width = data.width(), height = data.height(), channels = data.channels(), output_data = data.mutable_cpu_data()] (int w, int h, int c, TUnderlyingType pixel) { output_data[width * height * c + width * h + w] = pixel; } } :
                    std::function{ [&, width = data.width(), height = data.height(), channels = data.channels(), output_data = data.mutable_cpu_data()] (int w, int h, int c, TUnderlyingType pixel) { output_data[channels * width * h + channels * w + c] = pixel; } };
            }

            template<typename TUnderlyingType, bool shared, typename... TArgs>
            inline static auto make_tensor_core(TArgs&& ... args)
            {
                if constexpr (shared)
                {
                    return std::make_shared<tensor<TUnderlyingType>>(std::forward<TArgs>(args)...);
                }
                else
                {
                    return tensor<TUnderlyingType>{ std::forward<TArgs>(args)... };
                }
            }
        };
    }
}
