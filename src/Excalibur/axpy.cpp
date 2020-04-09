#include "axpy.hpp"
#include <algorithm>

using namespace glasssix::memory;

namespace glasssix
{
	namespace excalibur
	{
		axpy::axpy(int device)
		{
			device_ = device;
		}

		axpy::~axpy()
		{
		}

		void axpy::Forward_cpu(const std::vector<std::shared_ptr<tensor<float>>> bottom, std::shared_ptr<tensor<float>>& top)
		{
			CHECK_EQ(bottom.size(), 2);
			scales_ = bottom[0];
			input_ = bottom[1];
			order_ = input_->order();
			CHECK_EQ((scales_->data_shape()).size(), 2);
			CHECK_EQ(scales_->data_shape()[0], input_->num());
			CHECK_EQ(scales_->data_shape()[1], input_->channels());

			if (top == nullptr)
			{
				top.reset(new tensor<float>(input_->data_shape(), device_, order_));
			}
			else
			{
				CHECK_EQ(input_->data_shape()[0], top->data_shape()[0]);
				CHECK_EQ(input_->data_shape()[1], top->data_shape()[1]);
				CHECK_EQ(input_->data_shape()[2], top->data_shape()[2]);
				CHECK_EQ(input_->data_shape()[3], top->data_shape()[3]);

				if (order_ != top->order())
				{
					LOG(ERROR) << "order should be consistent!!!";
					return;
				}

				if (device_ != top->device())
				{
					LOG(ERROR) << "device should be consistent!!!";
					return;
				}
			}

			int num = input_->data_shape()[0];
			const float *scales_data = scales_->cpu_data();
			const float *bottom_data = input_->cpu_data();
			float *top_data = top->mutable_cpu_data();

			if (order_ == NCHW)
			{
				int channels = input_->data_shape()[1];
				int spatial_dim = input_->count(2, 4);

				for (int n = 0; n < num; ++n)
				{
					for (int ch = 0; ch < channels; ch++)
					{
						int scale_offset = n * channels + ch;
						int data_offset = scale_offset * spatial_dim;
						cblas_saxpby(spatial_dim, scales_data[scale_offset], bottom_data + data_offset, 1, 1.0f, top_data + data_offset, 1);
					}
				}
			}
			else if(order_ == NHWC)
			{
				int channels = input_->data_shape()[3];
				int spatial_dim = input_->count(1, 3);

				for (int n = 0; n < num; ++n)
				{
					for (int ch = 0; ch < channels; ch++)
					{
						int scale_offset = n * channels + ch;
						for (int i = 0; i < spatial_dim; i++)
						{
							int data_offset = n * spatial_dim * channels + i * channels + ch;
							top_data[data_offset] += scales_data[scale_offset] * bottom_data[data_offset];
					    }
					}
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}
	}
}

