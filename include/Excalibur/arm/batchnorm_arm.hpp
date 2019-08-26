#ifndef _BATCHNORM_ARM_HPP_
#define _BATCHNORM_ARM_HPP_

#include <glasssix/tensor.hpp>

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
				a_.reset(new tensor<float>(std::vector<int>{1, input_channel_, 1, 1}, -1, NCHW));
				b_.reset(new tensor<float>(std::vector<int>{1, input_channel_, 1, 1}, -1, NCHW));
			}

			void Forward_cpu(std::shared_ptr<tensor<float> >& bottom);
			void set_bias(float* bias)
			{
				b_->set_cpu_data(bias);
			}

			void set_weights(float* weights)
			{
				a_->set_cpu_data(weights);
			}

		private:
			//value = a * value + b;
			int input_channel_;
			std::shared_ptr<tensor<float> > a_;
			std::shared_ptr<tensor<float> > b_;
		};
	}
}

#endif