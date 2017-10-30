#pragma once
#ifndef _PCA_HPP_
#define _PCA_HPP_

#include "tensor.hpp"
#include "math_functions.hpp"

namespace excalibur
{
	class pca
	{
		tensor* weights_;
		int initial_dimensions;
		int final_dimensions;
		int device_;

	public:
		pca(int d, int k, int device);
		~pca();
		void set_weights(float* weights);
		void Forward_cpu(const std::shared_ptr<tensor>& bottom, std::shared_ptr<tensor>& top);
#ifdef USE_CUDA
		void Forward_native_gpu(cublasHandle_t cublas_handle_, const std::shared_ptr<tensor>& bottom, std::shared_ptr<tensor>& top);
#endif
	};
}

#endif