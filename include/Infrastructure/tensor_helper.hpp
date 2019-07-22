#pragma once

#include <cassert>

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
            /// Create a tensor base on specified type and bitmap data accordingly.
            /// </summary>
            /// <param name="bitmap">The bitmap data</param>
            /// <param name="order">The memory order</param>
            /// <param name="device">The device ID</param>
            /// <param name="width">The width in pixels</param>
            /// <param name="height">The height in pixels</param>
            /// <param name="stride">The number of bytes across a line</param>
            /// <param name="channels">The channel count</param>
            /// <returns>The result</returns>
            template<typename TUnderlyingType>
            static tensor<TUnderlyingType> create(const void* bitmap, orderType order, int device, int width, int height, int stride, int channels)
            {
                assert(bitmap != nullptr);
                static_assert(std::is_integral_v<TUnderlyingType> || std::is_floating_point_v<TUnderlyingType>, "The underlying type of a tensor must be integral or floating-point.");

                // Initialize necessary constants.
                auto channel_bytes = sizeof(TUnderlyingType);
                auto pixel_bytes = channel_bytes * channels;
                auto line_bytes = width * pixel_bytes;
                auto padding_bytes = stride - width * pixel_bytes;
                auto input_data = reinterpret_cast<const uint8_t*>(bitmap);
                auto input_vector = order == NHWC ? std::vector<int>{ 1, height, width, channels } : std::vector<int>{ 1, channels, height, width };

                // Create a tensor according to the memory order.
                tensor<TUnderlyingType> result{ input_vector, device, order };

                // Copy the data to the tensor.
                auto output_data = result.mutable_cpu_data();

                switch (order)
                {
                case NCHW:
                {
                    for (auto c = 0; c < channels; c++)
                    {
                        for (auto h = 0; h < height; h++)
                        {
                            for (auto w = 0; w < width; w++)
                            {
                                output_data[width * height * c + width * h + w] = *reinterpret_cast<const TUnderlyingType*>(input_data + (channels * width * h + channels * w + c) * channel_bytes + padding_bytes * h);
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
                        memcpy(reinterpret_cast<uint8_t*>(output_data) + index, input_data + index + padding_bytes * h, line_bytes);
                    }
                    break;
                }
                default:
                    break;
                }

                return result;
            }
        };
    }
}
