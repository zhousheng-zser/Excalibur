#include "fi_extended_conversions.hpp"
#include "FreeImage.h"

namespace glasssix
{
    namespace excalibur
    {
        const std::unordered_map<int, size_t> fi_extended_conversions::channel_byte_mapping_ =
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
            if (bitmap == nullptr)
            {
                return nullptr;
            }

            auto [data, width, height, stride] = get_parameters_core(bitmap);

            auto result = FreeImage_AllocateT(FIT_BITMAP, width, height);
        }

        /// <summary>
        /// Get the channel size of an image type.
        /// </summary>
        /// <param name="type">The image type</param>
        /// <returns>The channel size</returns>
        size_t fi_extended_conversions::channel_bytes(int type)
        {
            auto item = channel_byte_mapping_.find(type);

            return item != channel_byte_mapping_.end() ? item->second : 0;
        }

        std::tuple<uint8_t*, int, int, int> fi_extended_conversions::get_parameters_core(FIBITMAP* bitmap)
        {
            return { FreeImage_GetBits(bitmap), FreeImage_GetWidth(bitmap), FreeImage_GetHeight(bitmap), FreeImage_GetPitch(bitmap) };
        }
    }
}
