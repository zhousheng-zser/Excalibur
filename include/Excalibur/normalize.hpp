#pragma once
#ifndef _NORMALIZE_HPP_
#define _NORMALIZE_HPP_

#include "../../include/Primitives/tensor.hpp"
#include "math_functions.hpp"

namespace glasssix 
{
	namespace excalibur
	{
		class normalize
		{
			std::shared_ptr<memory::tensor<float>> sum_multiplier_;
			std::shared_ptr<memory::tensor<float>> norm_;
			std::shared_ptr<memory::tensor<float>> squared_;
			bool rescale_;
			enum normalize_type { L1, L2 };
			normalize_type type_;
			int device_;
			memory::orderType order_;
		public:
			normalize(int type, bool rescale, int device);
			~normalize();
			void Forward_cpu(const std::shared_ptr<memory::tensor<float>>& bottom);
#ifdef USE_CUDA
			void Forward_gpu_native(const std::shared_ptr<memory::tensor<float>>& bottom);
#endif
		};
	}
}


#endif