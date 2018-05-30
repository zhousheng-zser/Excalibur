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

			static void rotate_cpu(Tensor^ src, Tensor^ %dst, float theta,
				int center_x, int center_y, InterpolationType type, float v, int device);

		private:
			static void resize_cpu(Tensor^ src, Tensor^ %dst, int new_height, int new_width, InterpolationType type);

			static void rotate_cpu(Tensor^ src, Tensor^ %dst, float theta,
				int center_x, int center_y, InterpolationType type, float v);
		};
	}
}
#endif // !_TENSOR_CV_HPP_
