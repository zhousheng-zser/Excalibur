#ifndef _TENSOR_CV_HPP_
#define _TENSOR_CV_HPP_

#include "Tensor.hpp"
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

		private:
			static void resize_cpu(Tensor^ src, Tensor^ %dst, int new_height, int new_width, InterpolationType type);

			static void rotate_cpu(Tensor^ src, Tensor^ %dst, float theta,
				int center_x, int center_y, InterpolationType type, float v);

			static void flip_cpu(Tensor^ src, Tensor^ %dst, FlipType axis);

			static void rgb2gray_cpu(Tensor^ src, Tensor^ %dst);

			static void copy_make_border_cpu(Tensor^ src, Tensor^ %dst, int top, int bottom,
				int left, int right, BorderType type, float v);

			static void copy_cut_border_cpu(Tensor^ src, Tensor^ %dst, int top, int bottom, int left, int right);
		};
	}
}
#endif // !_TENSOR_CV_HPP_
