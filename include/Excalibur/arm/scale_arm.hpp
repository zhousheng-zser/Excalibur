#ifndef _SCALE_ARM_HPP_
#define _SCALE_ARM_HPP_
#include "../../Primitives/tensor.hpp"

namespace glasssix
{
	namespace excalibur
	{
		class scale_arm
		{
		public:
			scale_arm(int input_channel, bool bias_term) : input_channel_(input_channel), bias_term_(bias_term)
			{
				weights_.reset(new memory::tensor<float>(std::vector<int>{1, input_channel_, 1, 1}, -1, memory::NCHW));
				bias_.reset(new memory::tensor<float>(std::vector<int>{1, input_channel_, 1, 1}, -1, memory::NCHW));
			}

			void Forward_cpu(const std::shared_ptr<memory::tensor<float> >& bottom);
			void set_bias(float* bias)
			{
				bias_->set_cpu_data(bias);
			}

			void set_weights(float* weights)
			{
				weights_->set_cpu_data(weights);
			}

		private:
			int input_channel_;
			bool bias_term_;

			std::shared_ptr<memory::tensor<float> > weights_;
			std::shared_ptr<memory::tensor<float> > bias_;
		};
	}
}

#endif // _SCALE_ARM_HPP_
