#pragma once
#ifndef _PRELU_HPP_
#define _PRELU_HPP_
#include "tensor.hpp"
#include <memory>

namespace excalibur
{
	class prelu
	{
		tensor* slope_data_;
		bool isrelu_;
		int channel_;
		int device_;

	public:
		prelu(int input_channel, bool isrelu = false, int device = -1);

		~prelu();

		void setslope(float* slope_data);

		void Forward_cpu(const std::shared_ptr<tensor>& bottom);
#ifdef USE_CUDA
		void Forward_native_gpu(const std::shared_ptr<tensor>& bottom);
#endif
	};
}

#endif