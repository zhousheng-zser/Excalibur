#pragma once
#ifndef _PRELU_HPP_
#define _PRELU_HPP_
#include <glasssix\tensor.hpp>
#include <memory>

namespace glasssix
{
	namespace excalibur
	{
		class prelu
		{
			std::shared_ptr<tensor<float>> slope_data_;
			bool isrelu_;
			bool is_shared_;
			int channel_;
			int height_;
			int width_;
			int device_;
			orderType order_;

		public:
			prelu(int input_channel, bool isrelu = false, int device = -1, bool is_shared = false);

			~prelu();

			void setslope(float* slope_data);

			void Forward_cpu(const std::shared_ptr<tensor<float>>& bottom);
#ifdef USE_CUDA
			void Forward_gpu_native(const std::shared_ptr<tensor<float>>& bottom);
#endif
		};
	}
}


#endif