#pragma once
#ifndef _PRELU_HPP_
#define _PRELU_HPP_
#include "../../include/Primitives/tensor.hpp"
#include <memory>

namespace glasssix
{
	namespace excalibur
	{
		class prelu
		{
		protected:
			std::shared_ptr<memory::tensor<float>> slope_data_;
			bool isrelu_;
			bool is_shared_;
			int channel_;
			int height_;
			int width_;
			int device_;
			memory::orderType order_;

		public:
			prelu(int input_channel, bool isrelu = false, int device = -1, bool is_shared = false);

			virtual ~prelu();

			void setslope(float* slope_data);

			virtual void Forward_cpu(const std::shared_ptr<memory::tensor<float>>& bottom);
#ifdef USE_CUDA
			void Forward_gpu_native(const std::shared_ptr<memory::tensor<float>>& bottom);
#endif
		};
	}
}


#endif