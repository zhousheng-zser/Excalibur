#pragma once
#ifndef _PCA_HPP_
#define _PCA_HPP_

#include <glasssix/tensor.hpp>
#include "math_functions.hpp"

namespace glasssix
{
	namespace excalibur
	{
		class pca
		{
			std::shared_ptr<tensor<float>> weights_;
			int initial_dimensions;
			int final_dimensions;
			int device_;

		public:
			pca(int d, int k, int device);
			~pca();
			void set_weights(float* weights);
			void Forward_cpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);
#ifdef USE_CUDA
			void Forward_gpu_native(cublasHandle_t &cublas_handle_, const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);
#endif
		};
	}
}


#endif