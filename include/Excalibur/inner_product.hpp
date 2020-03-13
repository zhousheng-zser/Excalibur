#pragma once
#ifndef _INNER_PRODUCT_HPP_
#define _INNER_PRODUCT_HPP_
#include <glasssix/tensor.hpp>
#include "math_functions.hpp"
#include "im2col.hpp"

namespace glasssix
{
	namespace excalibur
	{
		class inner_product
		{
		protected:
			bool bias_term_;
			int num_output_;
			std::shared_ptr<tensor<float>> weights_;
			std::shared_ptr<tensor<float>> bias_;
			std::shared_ptr<tensor<float>> bias_multiplier_;
			std::vector<int> input_shape_without_num_;
			int K_;
			int N_;
			int M_;
			int device_;
			orderType order_;
		public:
			inner_product(std::vector<int> input_shape_withpout_num, int num_output, bool bias_term, int device);

			virtual ~inner_product();

			void set_weights(float* weights);

			void set_bias(float* bias);

			virtual void Forward_cpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);
#ifdef USE_CUDA
			void Forward_gpu_native(cublasHandle_t &cublas_handle_, const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);
#endif
		};
	}
}


#endif