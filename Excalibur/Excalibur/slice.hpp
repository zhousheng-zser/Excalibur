#pragma once
#ifndef _SLICE_HPP_
#define _SLICE_HPP_

#include "tensor.hpp"
#include "math_functions.hpp"

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

		void Forward_cpu(const tensor* bottom, std::vector<std::shared_ptr<tensor>>& top);
#ifdef USE_CUDA
		void Forward_native_gpu(const std::shared_ptr<tensor>& bottom, std::vector<std::shared_ptr<tensor>>& top);
#endif
	};
}

#endif