#ifndef _BATCHNORM_ARM_HPP_
#define _BATCHNORM_ARM_HPP_

#include "../../Primitives/tensor.hpp"

namespace glasssix
{
	namespace excalibur
	{
		class batchnorm_arm
		{
		public:
			batchnorm_arm(int input_channel) 
			{
				input_channel_ = input_channel;
				a_.reset(new memory::tensor<float>(std::vector<int>{1, input_channel_, 1, 1}, -1, memory::NCHW));
				b_.reset(new memory::tensor<float>(std::vector<int>{1, input_channel_, 1, 1}, -1, memory::NCHW));
			}

			void Forward_cpu(const std::shared_ptr<memory::tensor<float> >& bottom);
			void set_bias(float* bias)
			{
				//b_->set_cpu_data(bias);
				memcpy(b_->mutable_cpu_data(), bias, b_->count() * sizeof(float));
			}

			void set_weights(float* weights)
			{
				//a_->set_cpu_data(weights);
				memcpy(a_->mutable_cpu_data(), weights, a_->count() * sizeof(float));
			}

		private:
			//value = a * value + b;
			int input_channel_;
			std::shared_ptr<memory::tensor<float> > a_;
			std::shared_ptr<memory::tensor<float> > b_;
		};
	}
}

#endif