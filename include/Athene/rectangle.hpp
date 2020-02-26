#pragma once

#include "rgba_color.hpp"

#include <library_export_control.hpp>

namespace glasssix
{
	namespace ozymandias
	{
		/// <summary>
		/// Represent a rectangle.
		/// </summary>
		struct SPHINX_LIBRARY_API rectangle
		{
			/// <summary>
			/// The X of the left-top point.
			/// </summary>
			float left;

			/// <summary>
			/// The Y of the left-top point.
			/// </summary>
			float top;

			/// <summary>
			/// The X of the right-bottom point.
			/// </summary>
			float right;

			/// <summary>
			/// The Y of the right-bottom point.
			/// </summary>
			float bottom;

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

			rectangle(float left, float top, float right, float bottom);
			rectangle(float left, float top, float right, float bottom, float stroke_width);
			rectangle(float left, float top, float right, float bottom, float stroke_width, const rgba_color& stroke_color);
			rectangle(float left, float top, float right, float bottom, const rgba_color& stroke_color);
			rectangle(float left, float top, float right, float bottom, const rgba_color& stroke_color, const rgba_color& fill_color);
			rectangle(float left, float top, float right, float bottom, float stroke_width, const rgba_color& stroke_color, const rgba_color& fill_color);

			/// <summary>
			/// Get the width.
			/// </summary>
			/// <returns>The width</returns>
			float width() const;

			/// <summary>
			/// Get the height.
			/// </summary>
			/// <returns>The height</returns>
			float height() const;
		};
	}
}
