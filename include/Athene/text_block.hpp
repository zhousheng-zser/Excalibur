#pragma once

#include "rgba_color.hpp"
#include "simple_wstring.hpp"

#include <library_export_control.hpp>

namespace glasssix
{
    namespace ozymandias
    {
        /// <summary>
        /// Represent a text block.
        /// </summary>
        struct SPHINX_LIBRARY_API text_block
        {
			/// <summary>
			/// The X of the left-top point.
			/// </summary>
			float x;

			/// <summary>
			/// The Y of the left-top point;
			/// </summary>
			float y;

			/// <summary>
			/// The font color.
			/// </summary>
			rgba_color color;
			
			/// <summary>
			/// The text.
			/// </summary>
			simple_wstring text;

			text_block(const wchar_t* text, float x, float y);
			text_block(const wchar_t* text, float x, float y, const rgba_color& color);
        };
    }
}
