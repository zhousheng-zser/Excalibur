#pragma once

#include "rgba_color.hpp"

#include <library_export_control.hpp>

namespace glasssix
{
	namespace ozymandias
	{
		/// <summary>
		/// Represent an ellipse.
		/// </summary>
		struct SPHINX_LIBRARY_API ellipse
		{
			/// <summary>
			/// The X of the center point.
			/// </summary>
			float x;

			/// <summary>
			/// The Y of the center point.
			/// </summary>
			float y;

			/// <summary>
			/// The X radius.
			/// </summary>
			float radius_x;

			/// <summary>
			/// The Y radius.
			/// </summary>
			float radius_y;

			/// <summary>
			/// The stroke width.
			/// </summary>
			float stroke_width;

			/// <summary>
			/// The fill color.
			/// </summary>
			rgba_color fill_color;

			/// <summary>
			/// The stroke color.
			/// </summary>
			rgba_color stroke_color;

			ellipse(float x, float y, float radius_x, float radius_y);
			ellipse(float x, float y, float radius_x, float radius_y, float stroke_width);
			ellipse(float x, float y, float radius_x, float radius_y, float stroke_width, const rgba_color& stroke_color);
			ellipse(float x, float y, float radius_x, float radius_y, const rgba_color& stroke_color);
			ellipse(float x, float y, float radius_x, float radius_y, const rgba_color& stroke_color, const rgba_color& fill_color);
			ellipse(float x, float y, float radius_x, float radius_y, float stroke_width, const rgba_color& stroke_color, const rgba_color& fill_color);
		};
	}
}
