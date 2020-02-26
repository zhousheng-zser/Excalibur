#pragma once
#ifndef _UPSAMPLE_HPP_
#define _UPSAMPLE_HPP_
#include <glasssix/tensor.hpp>
#include <memory>

namespace glasssix
{
	namespace excalibur
	{
		class upsample
		{
		protected:
			int scale_;
			int channel_;
			int height_;
			int width_;
			int device_;
			orderType order_;

		public:
			upsample(int scale, int device = -1);

			virtual ~upsample();

			virtual void Forward_cpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);
#ifdef USE_CUDA
			void Forward_gpu_native(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);
#endif
		};
	}
}


#endif