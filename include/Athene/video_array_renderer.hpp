#pragma once

#include "video_view_decorator.hpp"

#include <Windows.h>
#include <library_export_control.hpp>

namespace glasssix
{
	struct bitmap_data_provider;

	namespace ozymandias
	{
		class video_array_renderer_impl;

		/// <summary>
		/// The video array renderer interface.
		/// </summary>
		class SPHINX_LIBRARY_API video_array_renderer
		{
		public:
			video_array_renderer(HWND window_handle);
			video_array_renderer(HWND window_handle, int margin);
			virtual ~video_array_renderer();

			void dismiss(int row, int column);
			void set_array(int rows, int columns);
			void switch_to_array_view();
			void switch_to_single_view(int row, int column);
			void set_single_background(HBITMAP bitmap);
			void set_single_background(HBITMAP bitmap, int row, int column);
			video_view_decorator get_view_decorator(int row, int column);
			void set_data_provider(bitmap_data_provider& provider, int row, int column);
		private:
			video_array_renderer_impl* impl_;
		};
	}
}
