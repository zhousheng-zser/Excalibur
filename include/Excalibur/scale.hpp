#pragma once
#ifndef _SCALE_HPP_
#define _SCALE_HPP_
#include "../../include/Primitives/tensor.hpp"
//using namespace glasssix::memory;

namespace glasssix
{
	namespace excalibur
	{
		class scale
		{
		protected:
			std::shared_ptr<glasssix::memory::tensor<float>> scales_data_, bias_data_;
			int num_;
			int channel_;
			int height_;
			int width_;
			int spatial_dim_;
			int bottom_dim_;
			int device_;
			glasssix::memory::orderType order_;

		public:
			scale(int input_channel, int device = -1);

			virtual ~scale();

			virtual void Forward_cpu(const std::shared_ptr<glasssix::memory::tensor<float>>& bottom);
#ifdef USE_CUDA
			void Forward_gpu_native(const std::shared_ptr<glasssix::memory::tensor<float>>& bottom);
#endif
		};
	}
}


#endif
