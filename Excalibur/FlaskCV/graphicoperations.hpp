#pragma once
#ifndef _GRAPHICOPERATIONS_HPP_
#define _GRAPHICOPERATIONS_HPP_

#include "graphic.hpp"
#include "rect.hpp"

namespace flaskcv
{
	class graphicoperations
	{
		static void bilinear_resize_cpu(std::shared_ptr<graphic> src, std::shared_ptr<graphic>& dst, int new_height, int new_width);

		static void flip_cpu(std::shared_ptr<graphic> src, std::shared_ptr<graphic>& dst, bool flip_height, bool flip_width);

		static void rgb2gray_cpu(std::shared_ptr<graphic> src, std::shared_ptr<graphic>& dst);

		static void copy_make_border_image_cpu(std::shared_ptr<graphic> src, std::shared_ptr<graphic>& dst, int top, int left, int type, float v);

		static void copy_make_border_cpu(std::shared_ptr<graphic> src, std::shared_ptr<graphic>& dst, int top, int bottom, int left, int right, int type, float v);

		static void copy_cut_border_image_cpu(std::shared_ptr<graphic> src, std::shared_ptr<graphic>& dst, int top, int left);

		static void copy_cut_border_cpu(std::shared_ptr<graphic> src, std::shared_ptr<graphic>& dst, int top, int bottom, int left, int right);

		enum bordertype{ BORDER_CONSTANT , BORDER_REPLICATE};
	public:
		graphicoperations();
		~graphicoperations();
	};
}


#endif _GRAPHICOPERATIONS_HPP_