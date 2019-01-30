#pragma once
#ifndef _CONCAT_HPP_
#define _CONCAT_HPP_

#include <glasssix\tensor.hpp>
#include "math_functions.hpp"

namespace glasssix
{
	namespace excalibur
	{
		class concat
		{
			int num_concats_;
			int concat_input_size_;
			int concat_axis_;

			int device_;
		public:
			concat(int concat_axis, int device);
			~concat();
			void Forward_cpu(const std::vector<std::shared_ptr<tensor<float>>> bottom, std::shared_ptr<tensor<float>>& top);

#ifdef USE_CUDA
			void Forward_native_gpu(const std::vector<std::shared_ptr<tensor<float>>> bottom, std::shared_ptr<tensor<float>>& top);
#endif
		};
	}
}


#endif // _CONCAT_HPP_