#pragma once

#include "line.hpp"
#include "ellipse.hpp"
#include "rectangle.hpp"
#include "text_block.hpp"

#include <library_export_control.hpp>

namespace std
{
	template<typename>
	class shared_ptr;
}

namespace glasssix
{
	namespace ozymandias
	{
		struct video_view_decorator_internal;

		/// <summary>
		/// The video view decorator interface.
		/// </summary>
		class SPHINX_LIBRARY_API video_view_decorator
		{
		public:
			using begin_init_handler_type = void(*)(void* any, video_view_decorator& context);
		public:
			video_view_decorator(const std::shared_ptr<video_view_decorator_internal>& decorator);
			virtual ~video_view_decorator();

			void clear();
			void add_line(const line& line);
			void add_text(const text_block& text);
			void add_ellipse(const ellipse& ellipse);
			void add_rectangle(const rectangle& rectangle);
			void begin_init(begin_init_handler_type handler, void* any = nullptr);
		private:
			std::shared_ptr<video_view_decorator_internal>* shared_impl_;
		};
	}
}
