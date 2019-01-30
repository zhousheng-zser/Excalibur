#include "prelu.hpp"
#include <memory>
#include <algorithm>

namespace glasssix
{
	namespace excalibur
	{
		prelu::prelu(int input_channel, bool isrelu, int device)
		{
			channel_ = input_channel;
			slope_data_.reset(new tensor<float>(std::vector<int>{input_channel}, device));
			isrelu_ = isrelu;
			device_ = device;
		}

		prelu::~prelu()
		{

		}

		void prelu::setslope(float* slope_data)
		{
			if (isrelu_)
			{
				memset(slope_data_->mutable_cpu_data(), 0, channel_ * sizeof(float));
			}
			else
			{
				for (int i = 0; i < channel_; i++)
				{
					slope_data_->mutable_cpu_data()[i] = slope_data[i];
				}
			}
		}

		void prelu::Forward_cpu(const std::shared_ptr<tensor<float>>& bottom)
		{
			int num = (bottom)->data_shape()[0];
			float* bottom_data = (bottom)->mutable_cpu_data();
			for (int i = 0; i < num; ++i)
			{
				CHECK_EQ(channel_, (bottom)->data_shape()[1]);
				for (int j = 0; j < channel_; j++)
				{
					const float slop = slope_data_->cpu_data()[j];
					int dim;
					if (bottom->data_shape().size() <= 2)
					{
						dim = 1;
					}
					else
					{
						dim = (bottom)->height()*(bottom)->width();
					}

					for (int k = 0; k < dim; k++)
					{
						*bottom_data = std::max(*bottom_data, 0.0f) + slop * std::min(*bottom_data, 0.0f);
						bottom_data++;
					}
				}
			}
		}


	}
}

