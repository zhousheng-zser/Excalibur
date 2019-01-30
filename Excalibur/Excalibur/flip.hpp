#pragma once
#ifndef _FLIP_HPP_
#define _FLIP_HPP_

#include <glasssix\tensor.hpp>
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
		public:
			flip(bool flip_height, bool flip_width, int device);
			~flip();
			void Forward_cpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);
#ifdef USE_CUDA
			void Forward_native_gpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);
#endif
		};
	}
}


#endif