#pragma once
#ifndef _SIGMOID_HPP_
#define _SIGMOID_HPP_
#include "tensor.hpp"
#include <memory>

namespace excalibur
{
	class sigmoid
	{
		int channel_;
		int device_;

	public:
		sigmoid(int input_channel, int device = -1);

		~sigmoid();

		void Forward_cpu(const std::shared_ptr<tensor<float>>& bottom);
#ifdef USE_CUDA
		void Forward_native_gpu(const std::shared_ptr<tensor<float>>& bottom);
#endif
	};
}

#endif