#include "../../include/Excalibur/operation_softmax.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"
#include "../../include/Excalibur/math_functions.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_softmax<Dtype>::operation_softmax(const operation_param& param) : operation<Dtype>(param)
		{
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					axis_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "1")
				{
					LOG(WARNING)<< "Un-supported Convolution Attribution " << split_string(attrs[i], "=")[0];
				}
				else if (split_string(attrs[i], "=")[0] == "-23330")
				{
					//do nothing
				}
				else
				{
					LOG(FATAL) << "Un-supported Convolution Attribution " << split_string(attrs[i], "=")[0];
				}
			}
		}

		template<typename Dtype>
		void operation_softmax<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), tops.size());
			for (size_t i = 0; i < bottoms.size(); i++)
			{
				int num = bottoms[i]->num();
				int channels = bottoms[i]->channels();
				int height = bottoms[i]->height();
				int width = bottoms[i]->width();
				int step;
				sum_multiplier_.reset(new memory::tensor<float>(channels, -1, memory::NCHW, nullptr));
				float* multiplier_data = sum_multiplier_->mutable_cpu_data();
				math_functions::cpu_set(channels, 1.0f, multiplier_data);
				if (bottoms[i]->data_shape().size() <= 2)
				{
					step = 1;
					this->scale_.reset(new memory::tensor<float>(std::vector<int>{num, 1}, -1, memory::NCHW, nullptr));
				}
				else
				{
					step = height * width;
					this->scale_.reset(new memory::tensor<float>(std::vector<int>{num, 1, height, width}, -1, memory::NCHW, nullptr));
				}

				tops[i].reset(new memory::tensor<float>(bottoms[i]->data_shape(), bottoms[i]->device(), bottoms[i]->order(), bottoms[i]->allocator()));
				//
				const float* bottom_data = bottoms[i]->cpu_data();
				float* top_data = tops[i]->mutable_cpu_data();
				float* scale_data = scale_->mutable_cpu_data();
				int dim = bottoms[i]->count(0, bottoms[i]->data_shape().size()) / num;
				memcpy(top_data, bottom_data, bottoms[i]->count(0, bottoms[i]->data_shape().size()) * sizeof(float));

				if (bottoms[i]->order() == memory::NCHW)
				{
					for (int n = 0; n < num; ++n)
					{
						memcpy(scale_data, bottom_data + n * dim, step * sizeof(float));
						for (int j = 0; j < channels; j++) 
						{
							for (int k = 0; k < step; k++) 
							{
								scale_data[k] = std::max(scale_data[k],
									bottom_data[n * dim + j * step + k]);
							}
						}
						math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, channels, step,
							1, -1.0f, sum_multiplier_->cpu_data(), scale_data, 1.0f, top_data);
						for (int k = 0; k < dim; k++)
						{
							top_data[k] = exp(top_data[k]);
						}
						math_functions::cpu_sgemv(CblasTrans, channels, step, 1.0f,
							top_data, sum_multiplier_->cpu_data(), 0.0f, scale_data);
						// division
						for (int j = 0; j < channels; j++) 
						{
							for (int k = 0; k < step; k++)
							{
								top_data[k] = top_data[k] / scale_data[k];
							}
							top_data += step;
						}
					}
				}
				else if (bottoms[i]->order() == memory::NHWC)
				{
					for (int n = 0; n < num; ++n)
					{
						memcpy(scale_data, bottom_data + n * dim, step * sizeof(float));

						//caculate max channel value, saved in scale_data
						for (int row = 0; row < height; row++)
						{
							int bottom_index1 = n * dim + row * width * channels;
							int scale_index1 = row * width;
							for (int col = 0; col < width; col++)
							{
								int bottom_index2 = bottom_index1 + col * channels;
								int scale_index2 = scale_index1 + col;
								for (int ch = 0; ch < channels; ch++)
								{
									scale_data[scale_index2] = std::max(scale_data[scale_index2],
										bottom_data[bottom_index2 + ch]);
								}
							}
						}

						//top_data subtract max value for each channel
						for (int row = 0; row < height; row++)
						{
							int top_index1 = n * dim + row * width * channels;
							int scale_index1 = row * width;
							for (int col = 0; col < width; col++)
							{
								int top_index2 = top_index1 + col * channels;
								int scale_index2 = scale_index1 + col;
								for (int ch = 0; ch < channels; ch++)
								{
									top_data[top_index2 + ch] -= scale_data[scale_index2];
								}
							}
						}

						//calculate exp for each top_data
						for (int k = 0; k < dim; k++)
						{
							top_data[k] = exp(top_data[k]);
						}

						//caculate channel sum, saved in scale_data
						for (int row = 0; row < height; row++)
						{
							int top_index1 = n * dim + row * width * channels;
							int scale_index1 = row * width;
							for (int col = 0; col < width; col++)
							{
								int top_index2 = top_index1 + col * channels;
								int scale_index2 = scale_index1 + col;
								float sum = 0.0f;
								for (int ch = 0; ch < channels; ch++)
								{
									sum += top_data[top_index2 + ch];
								}
								scale_data[scale_index2] = sum;
							}
						}

						// divide channel sum for each channel of top_data
						for (int row = 0; row < height; row++)
						{
							int top_index1 = n * dim + row * width * channels;
							int scale_index1 = row * width;
							for (int col = 0; col < width; col++)
							{
								int top_index2 = top_index1 + col * channels;
								int scale_index2 = scale_index1 + col;
								for (int ch = 0; ch < channels; ch++)
								{
									top_data[top_index2 + ch] = top_data[top_index2 + ch] / scale_data[scale_index2];
								}
							}
						}

						//offset
						top_data += dim;
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}
		}


		template<typename Dtype>
		void operation_softmax<Dtype>::forward_gpu_f32(
#ifdef USE_CUDA
			cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			NOT_IMPLEMENTED;
		}

		INSTANCE_CLASS(operation_softmax);
		REGISTE(operation_softmax);
	}
}