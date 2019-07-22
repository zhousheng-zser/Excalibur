#pragma once

#include <istream>
#include <optional>
#include <glasssix/tensor.hpp>

namespace glasssix
{
    namespace excalibur
    {
        /// <summary>
        /// Define supported floating bitmap types.
        /// </summary>
        enum class tensor_float_type
        {
            rgb,
            rgba,
            grayscale
        };

        /// <summary>
        /// Define supported uint8 bitmap types.
        /// </summary>
        enum class tensor_uint8_type
        {
            rgb_24bit,
            rgba_32bit,
            grayscale_8bit
        };

        /// <summary>
        /// Abstraction for a tensor builder.
        /// </summary>
        class tensor_builder
        {
        public:

            virtual ~tensor_builder() = default;

            /// <summary>
            /// Load a bitmap from a file.
            /// </summary>
            /// <param name="path">The path of the file</param>
            /// <returns>True on success</returns>
            virtual bool load_from(const std::string& path) = 0;

            /// <summary>
            /// Load a bitmap from an input stream.
            /// </summary>
            /// <param name="stream">The stream</param>
            /// <returns>True on success</returns>
            virtual bool load_from(std::istream& stream) = 0;

            /// <summary>
            /// Load a bitmap from a memory block.
            /// </summary>
            /// <param name="data">The memory block</param>
            /// <param name="size">The size in bytes</param>
            /// <returns>True on success</returns>
            virtual bool load_from(const void* data, size_t size) = 0;

            /// <summary>
            /// Save the image to a file.
            /// The encoder is deduced by the file extension automatically.
            /// </summary>
            /// <param name="path">The path of the file</param>
            /// <returns>True success</returns>
            virtual bool save_to(const std::string& path) = 0;

            /// <summary>
            /// Set the parameters for building a tensor.
            /// </summary>
            /// <param name="order">The memory order</param>
            virtual void tensor_parameters(orderType order) = 0;

            /// <summary>
            /// Set the parameters for building a tensor.
            /// </summary>
            /// <param name="order">The memory order</param>
            /// <param name="device">The device ID</param>
            virtual void tensor_parameters(orderType order, int device) = 0;

            /// <summary>
            /// Create a floating tensor.
            /// </summary>
            /// <param name="type">The destintation bitmap type</param>
            /// <returns>The result</returns>
            virtual std::optional<tensor<float>> to_tensor(tensor_float_type type) = 0;

            /// <summary>
            /// Create a uint8 tensor.
            /// </summary>
            /// <param name="type">The destination bitmap type</param>
            /// <returns>The result</returns>
            virtual std::optional<tensor<uint8_t>> to_tensor(tensor_uint8_type type) = 0;
        };
    }
}
