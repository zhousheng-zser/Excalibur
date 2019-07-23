#pragma once

#include "tensor_layout.hpp"

#include <istream>
#include <optional>

#include <glasssix/tensor.hpp>

namespace glasssix
{
    namespace excalibur
    {
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
            /// Create a bitmap from a floating-point tensor.
            /// </summary>
            /// <param name="data">The tensor data</param>
            /// <param name="layout">The tensor layout</param>
            /// <returns>
            /// True: success
            /// False: failure
            ///</returns>
            virtual bool from_tensor(const tensor<float>& data, tensor_layout layout) = 0;

            /// <summary>
            /// Create a bitmap from a uint8 tensor.
            /// </summary>
            /// <param name="data">The tensor data</param>
            /// <param name="layout">The tensor layout</param>
            /// <returns>
            /// True: success
            /// False: failure
            ///</returns>
            virtual bool from_tensor(const tensor<uint8_t>& data, tensor_layout layout) = 0;

            /// <summary>
            /// Create a floating tensor.
            /// </summary>
            /// <param name="layout">The destintation bitmap type</param>
            /// <returns>The result</returns>
            virtual std::optional<tensor<float>> to_tensor_float(tensor_layout layout) = 0;

            /// <summary>
            /// Create a uint8 tensor.
            /// </summary>
            /// <param name="layout">The destination bitmap type</param>
            /// <returns>The result</returns>
            virtual std::optional<tensor<uint8_t>> to_tensor_uint8(tensor_layout layout) = 0;

            /// <summary>
            /// Create a shared floating tensor.
            /// </summary>
            /// <param name="layout">The destintation bitmap type</param>
            /// <returns>The result</returns>
            virtual std::shared_ptr<tensor<float>> to_tensor_float_shared(tensor_layout layout) = 0;

            /// <summary>
            /// Create a shared uint8 tensor.
            /// </summary>
            /// <param name="layout">The destination bitmap type</param>
            /// <returns>The result</returns>
            virtual std::shared_ptr<tensor<uint8_t>> to_tensor_uint8_shared(tensor_layout layout) = 0;
        };
    }
}
