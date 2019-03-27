#include "prelu.hpp"
#include <memory>
#include <algorithm>
#include <iostream>

namespace glasssix
{
	namespace excalibur
	{
		prelu::prelu(int input_channel, bool isrelu, int device, bool is_shared)
		{
			channel_ = input_channel;
			isrelu_ = isrelu;
			is_shared_ = is_shared;
			device_ = device;
			if (!is_shared)
			{
				slope_data_.reset(new tensor<float>(std::vector<int>{input_channel}, device));
			}
			else
			{
				slope_data_.reset(new tensor<float>(std::vector<int>{1}, device));
			}
		}

		prelu::~prelu()
		{

		}

		void prelu::setslope(float* slope_data)
		{
			if (!is_shared_)
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
			else
			{
				if (isrelu_)
				{
					slope_data_->mutable_cpu_data()[0] = 0.0f;
				}
				else
				{
					slope_data_->mutable_cpu_data()[0] = slope_data[0];
				}
			}
		}

		void prelu::Forward_cpu(const std::shared_ptr<tensor<float>>& bottom)
		{
			int num = (bottom)->data_shape()[0];
			float* bottom_data = (bottom)->mutable_cpu_data();
			height_ = (bottom)->height();
			width_ = (bottom)->width();
			order_ = (bottom)->order();

			if (!is_shared_)
			{
				if (order_ == NCHW)
				{
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
				else if (order_ == NHWC)
				{
					CHECK_EQ(channel_, (bottom)->data_shape()[3]);

					for (int n = 0; n < num; ++n)
					{
						int index1 = n * height_ * width_ * channel_;
						for (int row = 0; row < (bottom)->height(); ++row)
						{
							int index2 = index1 + row * width_ * channel_;
							for (int col = 0; col < (bottom)->width(); ++col)
							{
								int index3 = index2 + col * channel_;
								for (int ch = 0; ch < channel_; ch++)
								{
									const float slop = slope_data_->cpu_data()[ch];
									bottom_data[index3 + ch] = std::max(bottom_data[index3 + ch], 0.0f) + slop * std::min(bottom_data[index3 + ch], 0.0f);
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
			else
			{
				const float slop = slope_data_->cpu_data()[0];
				if (order_ == NCHW)
				{
					for (int i = 0; i < num; ++i)
					{
						CHECK_EQ(channel_, (bottom)->data_shape()[1]);
						for (int j = 0; j < channel_; j++)
						{
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
				else if (order_ == NHWC)
				{
					CHECK_EQ(channel_, (bottom)->data_shape()[3]);

					for (int n = 0; n < num; ++n)
					{
						int index1 = n * height_ * width_ * channel_;
						for (int row = 0; row < (bottom)->height(); ++row)
						{
							int index2 = index1 + row * width_ * channel_;
							for (int col = 0; col < (bottom)->width(); ++col)
							{
								int index3 = index2 + col * channel_;
								for (int ch = 0; ch < channel_; ch++)
								{
									bottom_data[index3 + ch] = std::max(bottom_data[index3 + ch], 0.0f) + slop * std::min(bottom_data[index3 + ch], 0.0f);
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
}

