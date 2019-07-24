#include "fi_extended_conversions.hpp"
#include "FreeImage.h"

namespace glasssix
{
    namespace excalibur
    {
        const std::unordered_map<int, size_t> fi_extended_conversions::channel_bytes_mapping_ =
        {
            { FIT_BITMAP, sizeof(uint8_t) },
            { FIT_FLOAT, sizeof(float) },
            { FIT_DOUBLE, sizeof(double) },
            { FIT_RGBF, sizeof(float) },
            { FIT_RGBAF, sizeof(float) },
            { FIT_RGB16, sizeof(uint16_t) },
            { FIT_RGBA16, sizeof(uint16_t) }
        };

        /// <summary>
        /// Convert a RGBF image to a standard type.
        /// </summary>
        /// <param name="bitmap">The bitmap</param>
        /// <returns>The result</returns>
        FIBITMAP* fi_extended_conversions::rgbf_to_standard_type(FIBITMAP* bitmap)
        {
            return to_standard_type_core(bitmap, [](uint8_t* source, uint8_t* destination)
            {
                // RGBF: R-G-B
                // Standard RGB: B-G-R
                auto ptr = reinterpret_cast<float*>(source);
                *destination++ = static_cast<uint8_t>(*(ptr + 2) * 255.f);
                *destination++ = static_cast<uint8_t>(*(ptr + 1) * 255.f);
                *destination++ = static_cast<uint8_t>(*ptr * 255.f);
            });
        }

        /// <summary>
        /// Convert a RGBAF image to a standard type.
        /// </summary>
        /// <param name="bitmap">The bitmap</param>
        /// <returns>The result</returns>
        FIBITMAP* fi_extended_conversions::rgbaf_to_standard_type(FIBITMAP* bitmap)
        {
            return to_standard_type_core(bitmap, [](uint8_t* source, uint8_t* destination)
            {
                // RGBAF: R-G-B-A
                // Standard RGBA: B-G-R-A
                auto ptr = reinterpret_cast<float*>(source);
                *destination++ = static_cast<uint8_t>(*(ptr + 2) * 255.f);
                *destination++ = static_cast<uint8_t>(*(ptr + 1) * 255.f);
                *destination++ = static_cast<uint8_t>(*ptr * 255.f);
                *destination++ = static_cast<uint8_t>(*(ptr + 3) * 255.f);
            });
        }

        /// <summary>
        /// Convert an image to a standard type.
        /// </summary>
        /// <param name="bitmap">The bitmap</param>
        /// <returns>The result</returns>
        FIBITMAP* fi_extended_conversions::to_standard_type(FIBITMAP* bitmap)
        {
            // Invoke the standard conversion.
            auto result = FreeImage_ConvertToStandardType(bitmap);
            if (result != nullptr)
            {
                return result;
            }

            // If no support, invoke the extended conversion.
            switch (FreeImage_GetImageType(bitmap))
            {
            case FIT_RGBF:
                return fi_extended_conversions::rgbf_to_standard_type(bitmap);
            case FIT_RGBAF:
                return fi_extended_conversions::rgbaf_to_standard_type(bitmap);
            default:
                return nullptr;
            }
        }

        /// <summary>
        /// Get the channel size of an image type.
        /// </summary>
        /// <param name="type">The image type</param>
        /// <returns>The channel size</returns>
        size_t fi_extended_conversions::channel_bytes_of(int type)
        {
            auto item = channel_bytes_mapping_.find(type);

            return item != channel_bytes_mapping_.end() ? item->second : 0;
        }

        std::tuple<uint8_t*, int, int, int, int, int> fi_extended_conversions::get_parameters_core(FIBITMAP* bitmap)
        {
            return
            {
                FreeImage_GetBits(bitmap),
                static_cast<int>(FreeImage_GetWidth(bitmap)),
                static_cast<int>(FreeImage_GetHeight(bitmap)),
                static_cast<int>(FreeImage_GetPitch(bitmap)),
                static_cast<int>(FreeImage_GetBPP(bitmap)) / static_cast<int>(channel_bytes_of(FreeImage_GetImageType(bitmap))) / uint8_bits_,
                static_cast<int>(FreeImage_GetBPP(bitmap))
            };
        }

        FIBITMAP* fi_extended_conversions::to_standard_type_core(FIBITMAP* bitmap, const std::function<void(uint8_t*, uint8_t*)>& pixel_handler)
        {
            if (bitmap == nullptr || !pixel_handler)
            {
                return nullptr;
            }

            // Create a new bitmap.
            auto [data, width, height, stride, channels, pixel_bits] = get_parameters_core(bitmap);
            auto result = FreeImage_AllocateT(FIT_BITMAP, width, height, channels * uint8_bits_);

            if (result == nullptr)
            {
                return nullptr;
            }

            // Transform the bits.
            auto source_ptr = data;
            auto destination_ptr = FreeImage_GetBits(result);
            auto source_pixel_bytes = pixel_bits / uint8_bits_;
            auto source_padding_bytes = stride - width * source_pixel_bytes;
            auto destination_padding_bytes = FreeImage_GetPitch(result) - width * channels;

            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++, source_ptr += source_pixel_bytes, destination_ptr += channels)
                {
                    pixel_handler(source_ptr, destination_ptr);
                }

                source_ptr += source_padding_bytes;
                destination_ptr += destination_padding_bytes;
            }

            return result;
        }
    }
}
