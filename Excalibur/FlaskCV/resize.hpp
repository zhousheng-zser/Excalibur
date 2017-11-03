#pragma once
#ifndef _RESIZE_HPP_
#define _RESIZE_HPP_

#include "graphic.hpp"

namespace flaskcv
{
	class resize
	{
	public:
		resize();
		~resize();

		static void bilinear_resize(std::shared_ptr<graphic> src, std::shared_ptr<graphic>& dst, int new_height, int new_width);
	};
}

#endif