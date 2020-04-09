#pragma once
#ifndef _FLIP_HPP_
#define _FLIP_HPP_

#include "../../include/Primitives/tensor.hpp"
#include "math_functions.hpp"

namespace glasssix
{
	namespace excalibur
	{
		class flip
		{
			int device_;
			bool flip_height_;
			bool flip_width_;
			memory::orderType order_;
		public:
			flip(bool flip_height, bool flip_width, int device);
			~flip();
			void Forward_cpu(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top);
#ifdef USE_CUDA
			void Forward_gpu_native(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top);
#endif
		};
	}
}


#endif