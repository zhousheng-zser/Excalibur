#pragma once

#include "FreeImagePlus.h"

#include <optional>
#include <functional>

namespace glasssix
{
    namespace excalibur
    {
        /// <summary>
        /// An extension for fipImage.
        /// </summary>
        class fi_image_ex : public fipImage
        {
        public:
            using fipImage::fipImage;
            using fipImage::operator=;
        public:
            // Set an existing object.
            bool convert_to_4bits(fipImage& other);
            bool convert_to_8bits(fipImage& other);
            bool convert_to_grayscale(fipImage& other);
            bool convert_to_16bits_555(fipImage& other);
            bool convert_to_16bits_565(fipImage& other);
            bool convert_to_24bits(fipImage& other);
            bool convert_to_32bits(fipImage& other);
            bool convert_to_float(fipImage& other);
            bool convert_to_rgbf(fipImage& other);
            bool convert_to_rgbaf(fipImage& other);
            bool convert_to_uint16(fipImage& other);
            bool convert_to_rgb16(fipImage& other);
            bool convert_to_rgba16(fipImage& other);

            // Return an optional.
            std::optional<fi_image_ex> convert_to_4bits();
            std::optional<fi_image_ex> convert_to_8bits();
            std::optional<fi_image_ex> convert_to_grayscale();
            std::optional<fi_image_ex> convert_to_16bits_555();
            std::optional<fi_image_ex> convert_to_16bits_565();
            std::optional<fi_image_ex> convert_to_24bits();
            std::optional<fi_image_ex> convert_to_32bits();
            std::optional<fi_image_ex> convert_to_float();
            std::optional<fi_image_ex> convert_to_rgbf();
            std::optional<fi_image_ex> convert_to_rgbaf();
            std::optional<fi_image_ex> convert_to_uint16();
            std::optional<fi_image_ex> convert_to_rgb16();
            std::optional<fi_image_ex> convert_to_rgba16();

            // Extended functions.
            int width() const;
            int height() const;
            int stride() const;
            int channels() const;

            /// <summary>
            /// Convert the image to a standard type.
            /// </summary>
            /// <returns>
            /// True: success
            /// False: failure
            /// </returns>
            bool convert_to_standard_type();
        private:
            static FIBITMAP* rgbf_to_standard_type_core(FIBITMAP* bitmap);
            static FIBITMAP* rgbaf_to_standard_type_core(FIBITMAP* bitmap);
            bool convert_to_core(const std::function<FIBITMAP* (FIBITMAP*)>& handler, fipImage& other);
            std::optional<fi_image_ex> convert_to_core(const std::function<FIBITMAP* (FIBITMAP*)>& handler);
        private:
            static constexpr size_t uint8_bits_ = 8;
            static const std::unordered_map<int, std::function<FIBITMAP* (FIBITMAP*)>> extended_standard_type_converters_;
        };
    }
}
