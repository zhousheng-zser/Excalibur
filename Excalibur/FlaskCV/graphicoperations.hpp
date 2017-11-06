#pragma once
#ifndef _GRAPHICOPERATIONS_HPP_
#define _GRAPHICOPERATIONS_HPP_

#include "graphic.hpp"

namespace flaskcv
{
	class graphicoperations
	{
		static void bilinear_resize_cpu(std::shared_ptr<graphic> src, std::shared_ptr<graphic>& dst, int new_height, int new_width);

		static void flip_cpu(std::shared_ptr<graphic> src, std::shared_ptr<graphic>& dst, bool flip_height, bool flip_width);

		static void rgb2gray_cpu(std::shared_ptr<graphic> src, std::shared_ptr<graphic>& dst);
	public:
		graphicoperations();
		~graphicoperations();
	};
}


#endif _GRAPHICOPERATIONS_HPP_