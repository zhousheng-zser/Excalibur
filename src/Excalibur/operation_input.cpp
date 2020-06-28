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
						float means[] = { 104.f , 117.f, 123.f };
						//float means[] = { 0.0 , 0.0, 0.0 };
						float var = 1.0 / 128;
						//float var = 1;
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
												(bottom_data[offset + sub_offset + subsub_offset + w] - means[c]) * var;
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
												(bottom_data[offset + sub_offset + subsub_offset + c] - means[c]) * var;
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
						float var = 0.0078125f;
						for (int n = 0; n < num; n++)
						{
							int offset = n * height * width;
							for (int h = 0; h < height; h++)
							{
								int subsub_offset = h * width;
								for (int w = 0; w < width; w++)
								{
									top_data[offset + subsub_offset + w] =
										(bottom_data[offset + subsub_offset + w] - 127.5f) * var;
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