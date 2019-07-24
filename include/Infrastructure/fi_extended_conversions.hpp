#pragma once

#include <tuple>
#include <cstdint>
#include <functional>
#include <unordered_map>

struct FIBITMAP;

namespace glasssix
{
    namespace excalibur
    {
        /// <summary>
        /// Some extended conversions.
        /// </summary>
        class fi_extended_conversions final
        {
        public:
            /// <summary>
            /// Convert a RGBF image to a standard type.
            /// </summary>
            /// <param name="bitmap">The bitmap</param>
            /// <returns>The result</returns>
            static FIBITMAP* rgbf_to_standard_type(FIBITMAP* bitmap);

            /// <summary>
            /// Convert a RGBAF image to a standard type.
            /// </summary>
            /// <param name="bitmap">The bitmap</param>
            /// <returns>The result</returns>
            static FIBITMAP* rgbaf_to_standard_type(FIBITMAP* bitmap);

            /// <summary>
            /// Convert an image to a standard type.
            /// </summary>
            /// <param name="bitmap">The bitmap</param>
            /// <returns>The result</returns>
            static FIBITMAP* to_standard_type(FIBITMAP* bitmap);

            /// <summary>
            /// Get the channel size of an image type.
            /// </summary>
            /// <param name="type">The image type</param>
            /// <returns>The channel size</returns>
            static size_t channel_bytes_of(int type);
        private:
            static std::tuple<uint8_t*, int, int, int, int, int> get_parameters_core(FIBITMAP* bitmap);
            static FIBITMAP* to_standard_type_core(FIBITMAP* bitmap, const std::function<void(uint8_t*, uint8_t*)>& pixel_handler);
        private:
            static constexpr int uint8_bits_ = 8;
            static const std::unordered_map<int, size_t> channel_bytes_mapping_;
        };
    }
}
