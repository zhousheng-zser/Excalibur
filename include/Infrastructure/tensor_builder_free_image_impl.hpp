#pragma once

#include "tensor_helper.hpp"
#include "tensor_builder.hpp"

#include <tuple>
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>

class fipImage;

namespace glasssix
{
    namespace excalibur
    {
        class fi_image_ex;

        template<typename TEnum>
        using bitmap_converter_map = std::unordered_map<TEnum, std::function<std::optional<fi_image_ex>(fi_image_ex&)>>;

        /// <summary>
        /// FreeImage implementation.
        /// </summary>
        class tensor_builder_free_image_impl : public tensor_builder
        {
        public:
            tensor_builder_free_image_impl();
            virtual ~tensor_builder_free_image_impl();

            /// <summary>
           /// Load a bitmap from a file.
           /// </summary>
           /// <param name="path">The path of the file</param>
           /// <returns>True on success</returns>
            virtual bool load_from(const std::string& path) override;

            /// <summary>
            /// Load a bitmap from an input stream.
            /// </summary>
            /// <param name="stream">The stream</param>
            /// <returns>True on success</returns>
            virtual bool load_from(std::istream& stream) override;

            /// <summary>
            /// Load a bitmap from a memory block.
            /// </summary>
            /// <param name="data">The memory block</param>
            /// <param name="size">The size in bytes</param>
            /// <returns>True on success</returns>
            virtual bool load_from(const void* data, size_t size) override;

            /// <summary>
            /// Save the image to a file.
            /// The encoder is deduced by the file extension automatically.
            /// </summary>
            /// <param name="path">The path of the file</param>
            /// <returns>True success</returns>
            virtual bool save_to(const std::string& path) override;

            /// <summary>
            /// Set the parameters for building a tensor.
            /// </summary>
            /// <param name="order">The memory order</param>
            virtual void tensor_parameters(orderType order) override;

            /// <summary>
            /// Set the parameters for building a tensor.
            /// </summary>
            /// <param name="order">The memory order</param>
            /// <param name="device">The device ID</param>
            virtual void tensor_parameters(orderType order, int device) override;

            /// <summary>
            /// Create an image from a floating-point tensor.
            /// </summary>
            /// <param name="data">The tensor data</param>
            /// <param name="layout">The tensor layout</param>
            /// <returns>
            /// True: success
            /// False: failure
            ///</returns>
            virtual bool from_tensor(const tensor<float>& data, tensor_layout layout) override;

            /// <summary>
            /// Create a bitmap from a uint8 tensor.
            /// </summary>
            /// <param name="data">The tensor data</param>
            /// <param name="layout">The tensor layout</param>
            /// <returns>
            /// True: success
            /// False: failure
            ///</returns>
            virtual bool from_tensor(const tensor<uint8_t>& data, tensor_layout layout) override;

            /// <summary>
            /// Create a floating-point tensor.
            /// </summary>
            /// <param name="layout">The destintation bitmap type</param>
            /// <returns>The result</returns>
            virtual std::optional<tensor<float>> to_tensor_float(tensor_layout layout) override;

            /// <summary>
            /// Create a uint8 tensor.
            /// </summary>
            /// <param name="layout">The destination bitmap type</param>
            /// <returns>The result</returns>
            virtual std::optional<tensor<uint8_t>> to_tensor_uint8(tensor_layout layout) override;

            /// <summary>
            /// Create a shared floating-point tensor.
            /// </summary>
            /// <param name="layout">The destintation bitmap type</param>
            /// <returns>The result</returns>
            virtual std::shared_ptr<tensor<float>> to_tensor_float_shared(tensor_layout layout) override;

            /// <summary>
            /// Create a shared uint8 tensor.
            /// </summary>
            /// <param name="layout">The destination bitmap type</param>
            /// <returns>The result</returns>
            virtual std::shared_ptr<tensor<uint8_t>> to_tensor_uint8_shared(tensor_layout layout) override;
        private:
            /// <summary>
            /// Update the parameters of the image.
            /// </summary>
            void update_image_parameters_core();

            /// <summary>
            /// Convert a bitmap to a floating image type.
            /// Each pixel value is in the range [-1, 1].
            /// </summary>
            /// <param name="image">The image</param>
            /// <returns>True on success</returns>
            static bool convert_to_float_range_minus_one_to_one_core(fipImage& image);

            /// <summary>
            /// Convert the image to a tensor.
            /// </summary>
            /// <param name="type">The destintion bitmap type</param>
            /// <param name="converters">The converter cache</param>
            /// <returns>The result</returns>
            template<typename TEnum, typename TUnderlyingType, bool shared>
            auto to_tensor_core(TEnum type, bitmap_converter_map<TEnum>& converters)
                ->std::conditional_t<shared, std::shared_ptr<tensor<TUnderlyingType>>, std::optional<tensor<TUnderlyingType>>>;
        private:
            int width_;
            int height_;
            int stride_;
            int device_;
            int channels_;
            orderType order_;
            std::shared_ptr<fi_image_ex> image_;
        private:
            static constexpr int uint8_bits_ = 8;
            static bitmap_converter_map<tensor_layout> float_converters_;
            static bitmap_converter_map<tensor_layout> uint8_converters_;
        };
    }
}
