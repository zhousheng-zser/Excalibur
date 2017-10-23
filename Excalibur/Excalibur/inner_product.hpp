#pragma once
#ifndef _INNER_PRODUCT_HPP_
#define _INNER_PRODUCT_HPP_
#include "tensor.hpp"

namespace excalibur
{
	class inner_product
	{
		bool bias_term_;
		int num_output_;
		tensor* weights_;
		tensor* bias_;
		float* bias_multiplier_;
		std::vector<int> input_shape_without_num_;
		int K_;
		int N_;
		int M_;
		int device_;
	public:
		inner_product(std::vector<int> input_shape_withpout_num, int num_output, bool bias_term, int device);

		~inner_product();

		void set_weights(float* weights);

		void set_bias(float* bias);

		void Forward_cpu(const std::shared_ptr<tensor>& bottom, std::shared_ptr<tensor>& top);
	};
}

#endif