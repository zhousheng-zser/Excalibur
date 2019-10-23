#pragma once

#include <cstdint>

#include <library_export_control.hpp>

namespace glasssix
{
    namespace hippogriff
    {
		class ffmpeg_image;

        /// <summary>
        /// Store the decoded RGBA data.
        /// </summary>
        class SPHINX_LIBRARY_API sampling_data_info
        {
		public:
			sampling_data_info(const ffmpeg_image& image, bool duplicate = false);
			~sampling_data_info();

			int bytes() const;
			int width() const;
			int height() const;
			const uint8_t* data() const;
		private:
			int width_;
			int height_;
			int bytes_;
			uint8_t* data_;
			bool duplicated_;
        };
    }
}
