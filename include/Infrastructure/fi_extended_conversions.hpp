#pragma once

#include <tuple>
#include <cstdint>
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
            /// Get the channel size of an image type.
            /// </summary>
            /// <param name="type">The image type</param>
            /// <returns>The channel size</returns>
            static size_t channel_bytes(int type);
        private:
            static std::tuple<uint8_t*, int, int, int> get_parameters_core(FIBITMAP* bitmap);
        private:
            static const std::unordered_map<int, size_t> channel_byte_mapping_;
        };
    }
}
