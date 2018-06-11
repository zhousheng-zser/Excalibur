#ifndef _TENSOR_CV_HPP_
#define _TENSOR_CV_HPP_

#include "Tensor.hpp"
#include "UTensor.hpp"
#include "../Excalibur/tensoroperation.hpp"

namespace glasssix
{
	namespace gilgamesh
	{
		public ref class tensorcv
		{
		public:
			tensorcv() {}
			~tensorcv() {}

			static void resize(Tensor^ src, Tensor^ %dst, int new_height, int new_width, InterpolationType type, int device);

			static void rotate(Tensor^ src, Tensor^ %dst, float theta,
				int center_x, int center_y, InterpolationType type, float v, int device);

			static void flip(Tensor^ src, Tensor^ %dst, FlipType axis, int device);

			static void rgb2gray(Tensor^ src, Tensor^ %dst, int device);

			static void copy_make_border(Tensor^ src, Tensor^ %dst, int top, int bottom,
				int left, int right, BorderType type, float v, int device);

			static void copy_cut_border(Tensor^ src, Tensor^ %dst, int top, int bottom, int left, int right, int device);

			static void draw_rectangle(Tensor^ %dst, rectangle^ rect, int thickness, color^ color_, int device);

			static void equalize_hist(Tensor^ src, Tensor^ %dst, int device);

			static void lbp_feature(Tensor^ src, Tensor^ %dst, LbpType type, int device);

			static void mblbp_feature(Tensor^ src, Tensor^ %dst, int block_h, 
				int block_w, int stride_h, int stride_w, int device);

			static void safty_cut(Tensor^ src, Tensor^ %dst, rectangle^ rect, int device);
			//
			static void resize(UTensor^ src, UTensor^ %dst, int new_height, int new_width, InterpolationType type, int device);

			static void rotate(UTensor^ src, UTensor^ %dst, float theta,
				int center_x, int center_y, InterpolationType type, unsigned char v, int device);

			static void flip(UTensor^ src, UTensor^ %dst, FlipType axis, int device);

			static void rgb2gray(UTensor^ src, UTensor^ %dst, int device);

			static void copy_make_border(UTensor^ src, UTensor^ %dst, int top, int bottom,
				int left, int right, BorderType type, unsigned char v, int device);

			static void copy_cut_border(UTensor^ src, UTensor^ %dst, int top, int bottom, int left, int right, int device);

			static void draw_rectangle(UTensor^ %dst, rectangle^ rect, int thickness, color^ color_, int device);

			static void convert2UTensor(Tensor^ src, UTensor^ %dst);

			static void convert2Tensor(UTensor^ src, Tensor^ %dst);

			static void equalize_hist(UTensor^ src, UTensor^ %dst, int device);

			static void lbp_feature(UTensor^ src, UTensor^ %dst, LbpType type, int device);

			static void mblbp_feature(UTensor^ src, UTensor^ %dst, int block_h,
				int block_w, int stride_h, int stride_w, int device);
		private:
			static void resize_cpu(Tensor^ src, Tensor^ %dst, int new_height, int new_width, InterpolationType type);

			static void rotate_cpu(Tensor^ src, Tensor^ %dst, float theta,
				int center_x, int center_y, InterpolationType type, float v);

			static void flip_cpu(Tensor^ src, Tensor^ %dst, FlipType axis);

			static void rgb2gray_cpu(Tensor^ src, Tensor^ %dst);

			static void copy_make_border_cpu(Tensor^ src, Tensor^ %dst, int top, int bottom,
				int left, int right, BorderType type, float v);

			static void copy_cut_border_cpu(Tensor^ src, Tensor^ %dst, int top, int bottom, int left, int right);

			static void draw_rectangle_cpu(Tensor^ %dst, rectangle^ rect, int thickness, color^ color_);

			//
			static void resize_cpu(UTensor^ src, UTensor^ %dst, int new_height, int new_width, InterpolationType type);

			static void rotate_cpu(UTensor^ src, UTensor^ %dst, float theta,
				int center_x, int center_y, InterpolationType type, unsigned char v);

			static void flip_cpu(UTensor^ src, UTensor^ %dst, FlipType axis);

			static void rgb2gray_cpu(UTensor^ src, UTensor^ %dst);

			static void copy_make_border_cpu(UTensor^ src, UTensor^ %dst, int top, int bottom,
				int left, int right, BorderType type, unsigned char v);

			static void copy_cut_border_cpu(UTensor^ src, UTensor^ %dst, int top, int bottom, int left, int right);

			static void draw_rectangle_cpu(UTensor^ %dst, rectangle^ rect, int thickness, color^ color_);
		};
	}
}
#endif // !_TENSOR_CV_HPP_
