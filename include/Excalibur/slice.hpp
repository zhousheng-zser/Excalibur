#pragma once
#ifndef _SLICE_HPP_
#define _SLICE_HPP_

#include "../../include/Primitives/tensor.hpp"
#include "math_functions.hpp"

namespace glasssix
{
	namespace excalibur
	{
		class slice
		{
			int count_;
			int num_slices_;
			int slice_size_;
			int slice_axis_;
			std::vector<int> slice_point_;

			int device_;
		public:
			slice(int slice_axis, int device);

			~slice();

			void Forward_cpu(const std::shared_ptr<memory::tensor<float>> bottom, std::vector<std::shared_ptr<memory::tensor<float>>>& top);

			void Forward_cpu(const std::shared_ptr<memory::tensor<float>> bottom, std::shared_ptr<memory::tensor<float>>& top1, std::shared_ptr<memory::tensor<float>>& top2);
#ifdef USE_CUDA
			void Forward_gpu_native(const std::shared_ptr<memory::tensor<float>>& bottom, std::vector<std::shared_ptr<memory::tensor<float>>>& top);
#endif
		};
	}
}


#endif