#pragma once

#include "rgba_color.hpp"

#include <library_export_control.hpp>

namespace glasssix
{
	namespace ozymandias
	{
		/// <summary>
		/// Represent a line segment.
		/// </summary>
		struct SPHINX_LIBRARY_API line
		{
			/// <summary>
			/// The X of the first endpoint.
			/// </summary>
			float x1;

			/// <summary>
			/// The Y of the first endpoint.
			/// </summary>
			float y1;

			/// <summary>
			/// The X of the second endpoint.
			/// </summary>
			float x2;

			/// <summary>
			/// The Y of the second endpoint.
			/// </summary>
			float y2;

			/// <summary>
			/// The stroke width.
			/// </summary>
			float stroke_width;

			/// <summary>
			/// The stroke color.
			/// </summary>
			rgba_color stroke_color;

			line(float x1, float y1, float x2, float y2);
			line(float x1, float y1, float x2, float y2, float stroke_width);
			line(float x1, float y1, float x2, float y2, const rgba_color& stroke_color);
			line(float x1, float y1, float x2, float y2, float stroke_width, const rgba_color& stroke_color);
		};
	}
}
