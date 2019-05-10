#pragma once
#ifndef _AXPY_HPP_
#define _AXPY_HPP_
#include <glasssix\tensor.hpp>
#include "math_functions.hpp"


namespace glasssix
{
	namespace excalibur
	{
		class axpy
		{
			int device_;
			orderType order_;
			std::shared_ptr<tensor<float>> scales_;
			std::shared_ptr<tensor<float>> input_;

		public:
			axpy(int device);

			~axpy();

			void Forward_cpu(const std::vector<std::shared_ptr<tensor<float>>> bottom, std::shared_ptr<tensor<float>>& top);

#ifdef USE_CUDA
			void Forward_gpu_native(cublasHandle_t cublas_handle_, const std::vector<std::shared_ptr<tensor<float>>> bottom, std::shared_ptr<tensor<float>>& top);
#endif
		};
	}
}


#endif