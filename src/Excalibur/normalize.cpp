#include "normalize.hpp"

namespace glasssix
{
	namespace excalibur
	{
		normalize::normalize(int type, bool rescale, int device)
		{
			type_ = (normalize_type)type;
			rescale_ = rescale;
			device_ = device;
		}


		normalize::~normalize()
		{
		}

		void normalize::Forward_cpu(const std::shared_ptr<tensor<float>>& bottom)
		{
			int num = bottom->num();
			int channels = bottom->channels();
			int height = bottom->height();
			int width = bottom->width();
			order_ = bottom->order();

			if (order_ == NCHW)
			{
				squared_.reset(new tensor<float>(std::vector<int>{num, channels, height, width}, device_, order_));
				norm_.reset(new tensor<float>(std::vector<int>{num, 1, height, width}, device_, order_));
				//
				const float* bottom_data = bottom->cpu_data();
				float* top_data = bottom->mutable_cpu_data();
				float* square_data = squared_->mutable_cpu_data();
				float* norm_data = norm_->mutable_cpu_data();
				int spatial_dim = height * width;

				if (type_ == L2)
				{
					math_functions::cpu_sqr(num*channels*spatial_dim, bottom_data, square_data);
					for (int n = 0; n < num; n++) {
						for (int s = 0; s < spatial_dim; s++) {
							norm_data[n*spatial_dim + s] = 0.0f;
							for (int c = 0; c < channels; c++) {
								norm_data[n*spatial_dim + s] += square_data[(n * channels + c) * spatial_dim + s];
							}
							norm_data[n*spatial_dim + s] += 1e-6;
							norm_data[n*spatial_dim + s] = 1.0f / sqrt(norm_data[n*spatial_dim + s]);
							for (int c = 0; c < channels; c++) {
								top_data[(n * channels + c) * spatial_dim + s] = bottom_data[(n * channels + c) * spatial_dim + s] * norm_data[n*spatial_dim + s];
							}
						}
					}
				}
				else if (type_ == L1)
				{
					math_functions::cpu_abs(num*channels*spatial_dim, bottom_data, square_data);
					for (int n = 0; n < num; n++) {
						for (int s = 0; s < spatial_dim; s++) {
							norm_data[n*spatial_dim + s] = 0.0f;
							for (int c = 0; c < channels; c++) {
								norm_data[n*spatial_dim + s] += square_data[(n * channels + c) * spatial_dim + s];
							}
							norm_data[n*spatial_dim + s] += 1e-6;
							norm_data[n*spatial_dim + s] = 1.0f / norm_data[n*spatial_dim + s];
							for (int c = 0; c < channels; c++) {
								top_data[(n * channels + c) * spatial_dim + s] = bottom_data[(n * channels + c) * spatial_dim + s] * norm_data[n*spatial_dim + s];
							}
						}
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}
			else if (order_ == NHWC)
			{
				squared_.reset(new tensor<float>(std::vector<int>{num, height, width, channels}, device_, order_));
				norm_.reset(new tensor<float>(std::vector<int>{num, height, width, 1}, device_, order_));
				//
				const float* bottom_data = bottom->cpu_data();
				float* top_data = bottom->mutable_cpu_data();
				float* square_data = squared_->mutable_cpu_data();
				float* norm_data = norm_->mutable_cpu_data();
				int spatial_dim = height * width;

				if (type_ == L2)
				{
					math_functions::cpu_sqr(num*channels*spatial_dim, bottom_data, square_data);
					for (int n = 0; n < num; n++) 
					{
						int norm_index1 = n * height * width;
						int square_index1 = n * height * width * channels;
						for (int row = 0; row < height; row++)
						{
							int norm_index2 = norm_index1 + row * width;
							int square_index2 = square_index1 + row * width * channels;

							for (int col = 0; col < width; col++)
							{
								int norm_index3 = norm_index2 + col;
								int square_index3 = square_index2 + col * channels;

								norm_data[norm_index3] = 0.0f;
								for (int ch = 0; ch < channels; ch++)
								{
									norm_data[norm_index3] += square_data[square_index3 + ch];
								}

								norm_data[norm_index3] += 1e-6;
								norm_data[norm_index3] = 1.0f / sqrt(norm_data[norm_index3]);

								for (int ch = 0; ch < channels; ch++)
								{
									top_data[square_index3 + ch] = bottom_data[square_index3 + ch]  * norm_data[norm_index3];
								}
							}
						}
					}
				}
				else if (type_ == L1)
				{
					math_functions::cpu_abs(num*channels*spatial_dim, bottom_data, square_data);
					for (int n = 0; n < num; n++)
					{
						int norm_index1 = n * height * width;
						int square_index1 = n * height * width * channels;
						for (int row = 0; row < height; row++)
						{
							int norm_index2 = norm_index1 + row * width;
							int square_index2 = square_index1 + row * width * channels;

							for (int col = 0; col < width; col++)
							{
								int norm_index3 = norm_index2 + col;
								int square_index3 = square_index2 + col * channels;

								norm_data[norm_index3] = 0.0f;
								for (int ch = 0; ch < channels; ch++)
								{
									norm_data[norm_index3] += square_data[square_index3 + ch];
								}

								norm_data[norm_index3] += 1e-6;
								norm_data[norm_index3] = 1.0f / norm_data[norm_index3];

								for (int ch = 0; ch < channels; ch++)
								{
									top_data[square_index3 + ch] = bottom_data[square_index3 + ch] * norm_data[norm_index3];
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
				NOT_IMPLEMENTED;
			}
		}
	}
}

