#include "../../include/Excalibur/operation_input.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_input<Dtype>::operation_input(const operation_param& param) : operation<Dtype>(param)
		{
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					w_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "1")
				{
					h_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "2")
				{
					c_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "3")
				{
					auto means_str = split_string(split_string(attrs[i], "=")[1], ",");
					for (size_t j = 0; j < means_str.size(); j++)
					{
						means_.push_back(atof(means_str[j].c_str()));
					}
				}
				else if (split_string(attrs[i], "=")[0] == "4")
				{
					var_ = atof(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "-23330")
				{
					//do nothing
				}
				else
				{
					LOG(FATAL) << "Un-supported Input Attribution " << split_string(attrs[i], "=")[0];
				}
			}
		}

		template<typename Dtype>
		void operation_input<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), tops.size());
			for (size_t i = 0; i < bottoms.size(); i++)
			{
				tops[i].reset(new memory::tensor<float>(bottoms[i]->data_shape(), bottoms[i]->device(), bottoms[i]->order(), bottoms[i]->allocator()));
				int num = bottoms[i]->num();
				int channel = bottoms[i]->channels();
				int height = bottoms[i]->height();
				int width = bottoms[i]->width();
				int offset = height * width;
				int num_offset = channel * height * width;
				float* top_data = tops[i]->mutable_cpu_data();
				const float* bottom_data = bottoms[i]->cpu_data();
				for (size_t n = 0; n < num; n++)
				{
					if (channel == 3)
					{
						CHECK_EQ(channel, means_.size());
						if (bottoms[i]->order() == memory::NCHW)
						{
							for (int n = 0; n < num; n++)
							{
								int offset = n * channel * height * width;
								for (int c = 0; c < 3; c++)
								{
									int sub_offset = c * height * width;
									for (int h = 0; h < height; h++)
									{
										int subsub_offset = h * width;
										for (int w = 0; w < width; w++)
										{
											top_data[offset + sub_offset + subsub_offset + w] =
												(bottom_data[offset + sub_offset + subsub_offset + w] - means_[c]) * var_;
										}
									}
								}
							}
						}
						else if (bottoms[i]->order() == memory::NHWC)
						{
							for (int n = 0; n < num; n++)
							{
								int offset = n * 3 * height * width;
								for (int h = 0; h < height; h++)
								{
									int sub_offset = h * width * 3;
									for (int w = 0; w < width; w++)
									{
										int subsub_offset = w * 3;
										for (int c = 0; c < 3; c++)
										{
											top_data[offset + sub_offset + subsub_offset + c] =
												(bottom_data[offset + sub_offset + subsub_offset + c] - means_[c]) * var_;
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
					else if (channel == 1)
					{
						CHECK_EQ(channel, means_.size());
						for (int n = 0; n < num; n++)
						{
							int offset = n * height * width;
							for (int h = 0; h < height; h++)
							{
								int subsub_offset = h * width;
								for (int w = 0; w < width; w++)
								{
									top_data[offset + subsub_offset + w] =
										(bottom_data[offset + subsub_offset + w] - means_[0]) * var_;
								}
							}
						}
					}
					else
					{
						LOG(FATAL) << "Un-supprted channel num: " << channel;
					}
				}
			}
		}


		template<typename Dtype>
		void operation_input<Dtype>::forward_gpu_f32(
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

		INSTANCE_CLASS(operation_input);
		REGISTE(operation_input);
	}
}