#ifndef _TENSOROPERATION_HPP_
#define _TENSOROPERATION_HPP_

#include "tensor.hpp"

namespace excalibur
{
	template <typename Dtype>
	class tensoroperation
	{
#define Get_Index(x, y, offset) (y*offset+x) 
		enum bordertype { BORDER_CONSTANT, BORDER_REPLICATE };
	public:
		tensoroperation();
		~tensoroperation();

		static void bilinear_resize_cpu(std::shared_ptr<tensor<Dtype>> src, 
			std::shared_ptr<tensor<Dtype>>& dst, int new_height, int new_width);

		static void flip_cpu(std::shared_ptr<tensor<Dtype>> src, 
			std::shared_ptr<tensor<Dtype>>& dst, std::string axis);

		static void rgb2gray_cpu(std::shared_ptr<tensor<Dtype>> src, 
			std::shared_ptr<tensor<Dtype>>& dst);


		static void copy_make_border_cpu(std::shared_ptr<tensor<Dtype>> src, 
			std::shared_ptr<tensor<Dtype>>& dst, int top, int bottom, int left, int right, int type, Dtype v);

		static void copy_cut_border_cpu(std::shared_ptr<tensor<Dtype>> src, 
			std::shared_ptr<tensor<Dtype>>& dst, int top, int bottom, int left, int right);

	private:
		static void copy_make_border_image_cpu(std::shared_ptr<tensor<Dtype>> src,
			std::shared_ptr<tensor<Dtype>>& dst, int top, int left, int type, Dtype v);

		static void copy_cut_border_image_cpu(std::shared_ptr<tensor<Dtype>> src,
			std::shared_ptr<tensor<Dtype>>& dst, int top, int left);
	};
}
#endif // !_TENSOROPERATION_HPP_



