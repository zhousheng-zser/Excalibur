#pragma once

#include "well_known_color.hpp"

#include <library_export_control.hpp>

namespace glasssix
{
	namespace ozymandias
	{
		/// <summary>
		/// Represent a RGBA color.
		/// </summary>
		struct SPHINX_LIBRARY_API rgba_color
		{
			float red;
			float green;
			float blue;
			float alpha;

			rgba_color(bool transparent = false);
			rgba_color(float gray);
			rgba_color(float gray, float alpha);
			rgba_color(float red, float green, float blue);
			rgba_color(float red, float green, float blue, float alpha);
			rgba_color(well_known_color color);
			rgba_color(well_known_color color, float alpha);
		};
	}
}
