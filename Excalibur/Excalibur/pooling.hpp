#pragma once
#ifndef _POOLING_HPP_
#define _POOLING_HPP_
#include "tensor.hpp"
#ifdef USE_CUDNN
#include "cudnn.hpp"
#endif
namespace excalibur
{
	class pooling
	{
		int channels_;
		int height_, width_;
		int pooled_height_, pooled_width_;
		int kernel_;
		int stride_;
		int pad_;
		int device_;
		enum pooling_type { MAX, AVE };
		pooling_type type_;
#ifdef USE_CUDNN
		float one = 1.0, zero = 0.0;
		cudnnTensorDescriptor_t bottom_desc_, top_desc_;
		cudnnPoolingDescriptor_t  pooling_desc_;
		cudnnPoolingMode_t        mode_;
#endif
	public:
		pooling(int kernel, int stride, int pad, int type, int device);
		~pooling();

		void Forward_cpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);

#ifdef USE_CUDA
		void Forward_native_gpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);
#ifdef USE_CUDNN
		void Forward_cudnn_gpu(cudnnHandle_t cudnn_handle_, const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);
#endif
#endif
	};
}

#endif
