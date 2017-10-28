#pragma once
#ifndef _NORMALIZE_HPP_
#define _NORMALIZE_HPP_

#include "tensor.hpp"
#include "math_functions.hpp"

namespace excalibur
{
	class normalize
	{
		std::shared_ptr<tensor> sum_multiplier_;
		std::shared_ptr<tensor> norm_;
		std::shared_ptr<tensor> squared_;
		bool rescale_;
		enum normalize_type{L1, L2};
		normalize_type type_;
		int device_;
	public:
		normalize(int type, bool rescale, int device);
		~normalize();
		void Forward_cpu(const std::shared_ptr<tensor>& bottom);
#ifdef USE_CUDA
		void Forward_native_gpu(const std::shared_ptr<tensor>& bottom);
#endif
	};
}

#endif