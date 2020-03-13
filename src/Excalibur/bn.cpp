#include "scale.hpp"
//#include <memory>
#include <algorithm>
#include <cmath>
namespace glasssix
{
	namespace excalibur
	{
		scale::scale(int input_channel, int device)
		{
			channel_ = input_channel;
			device_ = device;
			scales_data_.reset(new tensor<float>(std::vector<int>{input_channel}, device));
			bias_data_.reset(new tensor<float>(std::vector<int>{input_channel}, device));
		}

		scale::~scale()
		{

		}


		void scale::Forward_cpu(const std::shared_ptr<tensor<float>>& bottom)
		{
			int bottom_channel = bottom->channels();
			CHECK_EQ(channel_, bottom_channel);
			num_ = bottom->num();
			order_ = bottom->order();
			height_ = bottom->height();
			width_ = bottom->width();
			spatial_dim_ = height_ * width_;
			bottom_dim_ = bottom->count(1, 4);
			float* bottom_data = bottom->mutable_cpu_data();
			const float *scales_data = scales_data_->cpu_data();
			const float *bias_data = bias_data_->cpu_data();

			if (order_ == NCHW)
			{
				for (int n = 0; n < num_; n++)
				{
					int offset_num = n * bottom_dim_;
					for (int ich = 0; ich < channel_; ich++)
					{
						float scale0 = scales_data[ich];
						float bias0 = bias_data[ich];
						int offset_num_channel = offset_num + ich * spatial_dim_;

						for (int row = 0; row < height_; row++)
						{
							int offset_num_channel_row = offset_num_channel + row * width_;
							for (int col = 0; col < width_; col++)
							{
								int offset_num_channel_row_col = offset_num_channel_row + col;
								bottom_data[offset_num_channel_row_col] = bottom_data[offset_num_channel_row_col] * scale0 + bias0;
							}
						}
					}
				}
			}
			else if(order_ == NHWC)
			{
				for (int n = 0; n < num_; n++)
				{
					int offset_num = n * bottom_dim_;
					for (int row = 0; row < height_; row++)
					{
						int offset_num_row = offset_num + row * width_ * channel_;
						for (int col = 0; col < width_; col++)
						{
							int offset_num_row_col = offset_num_row + col * channel_;
							for (int ich = 0; ich < channel_; ich++)
							{
								float scale0 = scales_data[ich];
								float bias0 = bias_data[ich];
								int offset_num_channel_row_col = offset_num_row_col + ich;
								bottom_data[offset_num_channel_row_col] = bottom_data[offset_num_channel_row_col] * scale0 + bias0;
							}							
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

