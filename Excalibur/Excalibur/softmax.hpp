#pragma once
#ifndef _SOFTMAX_HPP_
#define _SOFTMAX_HPP_
#include "tensor.hpp"
#include "math_functions.hpp"

namespace excalibur
{
	class softmax
	{
		int outer_num_;
		int inner_num_;
		int softmax_axis_;

		/// sum_multiplier is used to carry out sum using BLAS
		tensor* sum_multiplier_;
		/// scale is an intermediate Blob to hold temporary results.
		tensor* scale_;

		int device_;
	public:
		softmax(int input_channel, int device);
		~softmax();

		void Forward_cpu(const std::shared_ptr<tensor>& bottom, std::shared_ptr<tensor>& top);
	};
}

#endif