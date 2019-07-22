#pragma once

#include "tensor_helper.hpp"
#include "tensor_builder.hpp"

#include <atomic>
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>

#include <FreeImagePlus.h>

namespace glasssix
{
    namespace excalibur
    {
        template<typename TEnum>
        using bitmap_converter_map = std::unordered_map<TEnum, std::function<bool(fipImage&)>>;

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
            /// Create a floating tensor.
            /// </summary>
            /// <param name="type">The destintation bitmap type</param>
            /// <returns>The result</returns>
            virtual std::optional<tensor<float>> to_tensor(tensor_float_type type) override;

            /// <summary>
            /// Create a uint8 tensor.
            /// </summary>
            /// <param name="type">The destination bitmap type</param>
            /// <returns>The result</returns>
            virtual std::optional<tensor<uint8_t>> to_tensor(tensor_uint8_type type) override;
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
            template<typename TEnum, typename TUnderlyingType>
            std::optional<tensor<TUnderlyingType>> to_tensor_core(TEnum type, bitmap_converter_map<TEnum>& converters)
            {
                static_assert(std::is_enum_v<TEnum>, "The TEnum must be an enumeration type.");
                static_assert(std::is_integral_v<TUnderlyingType> || std::is_floating_point_v<TUnderlyingType>, "The underlying type of a tensor must be integral or floating-point.");

                if (!image_->isValid())
                {
                    std::nullopt;
                }

                // Find the appropriate floating converter.
                auto item = converters.find(type);
                if (item == converters.end() || !item->second(*image_))
                {
                    return std::nullopt;
                }

                update_image_parameters_core();

                return tensor_helper::create<TUnderlyingType>(image_->accessPixels(), order_, device_, width_, height_, stride_, channels_);
            }
        private:
            int width_;
            int height_;
            int stride_;
            int channels_;
            std::atomic_int device_;
            std::atomic<orderType> order_;
            std::shared_ptr<fipImage> image_;
        private:
            static constexpr int uint8_bits_ = 8;
            inline static std::unordered_map<FREE_IMAGE_TYPE, size_t> channel_byte_mapping_ =
            {
                { FREE_IMAGE_TYPE::FIT_BITMAP, sizeof(uint8_t) },
                { FREE_IMAGE_TYPE::FIT_FLOAT, sizeof(float) },
                { FREE_IMAGE_TYPE::FIT_DOUBLE, sizeof(double) },
                { FREE_IMAGE_TYPE::FIT_RGBF, sizeof(float) },
                { FREE_IMAGE_TYPE::FIT_RGBAF, sizeof(float) },
                { FREE_IMAGE_TYPE::FIT_RGB16, sizeof(uint16_t) },
                { FREE_IMAGE_TYPE::FIT_RGBA16, sizeof(uint16_t) }
            };

            inline static bitmap_converter_map<tensor_float_type> float_converters_ =
            {
                { tensor_float_type::rgb, [](fipImage& inner) { return inner.convertToRGBF(); } },
                { tensor_float_type::rgba, [](fipImage& inner) { return inner.convertToRGBAF(); } },
                { tensor_float_type::grayscale, [](fipImage& inner) { return inner.convertToFloat(); } }
            };

            inline static bitmap_converter_map<tensor_uint8_type> uint8_converters_ =
            {
                { tensor_uint8_type::rgb_24bit, [](fipImage& inner) { return inner.convertTo24Bits(); } },
                { tensor_uint8_type::rgba_32bit, [](fipImage& inner) { return inner.convertTo32Bits(); } },
                { tensor_uint8_type::grayscale_8bit, [](fipImage& inner) { return inner.convertToGrayscale(); } }
            };
        };
    }
}
