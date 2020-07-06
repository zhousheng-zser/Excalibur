#include "../../include/Excalibur/operation_prelu.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_prelu<Dtype>::operation_prelu(const operation_param& param) : operation<Dtype>(param)
		{
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					num_slope_ = atoi(split_string(attrs[i], "=")[1].c_str());
					CHECK_GE(num_slope_, 1);
					if (num_slope_ == 1)
					{
						share_channel_ = true;
					}
				}
				else if (split_string(attrs[i], "=")[0] == "-23330")
				{
					//do nothing
				}
				else
				{
					LOG(FATAL) << "Un-supported PReLU Attribution " << split_string(attrs[i], "=")[0];
				}
			}
			params_.inplace_ = true;
		}

		template<typename Dtype>
		int operation_prelu<Dtype>::init_weights(FILE *fp)
		{
			weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(num_slope_, params_.device_, memory::NCHW, nullptr)));
			fread(weights_f32_[0]->mutable_cpu_data(), 1, num_slope_ * sizeof(float), fp);
			return num_slope_ * sizeof(float);
		}

		template<typename Dtype>
		void operation_prelu<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), tops.size());
			for (size_t i = 0; i < bottoms.size(); i++)
			{
				tops[i].reset(new memory::tensor<float>(bottoms[i]->data_shape(), bottoms[i]->device(), bottoms[i]->order(), bottoms[i]->allocator()));
				float* top_data = tops[i]->mutable_cpu_data();
				const float* bottom_data = bottoms[i]->cpu_data();
				const float* slope_data = weights_f32_[0]->cpu_data();
				if (bottoms[i]->order() == memory::NCHW)
				{
					for (size_t n = 0; n < bottoms[i]->num(); n++)
					{
						int ch = bottoms[i]->channels();
						for (size_t c = 0; c < ch; c++)
						{
							int step = bottoms[i]->count(2, 4);
							int offset = n * ch * step + c * step;
							if (share_channel_)
							{
//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
								for (int j = 0; j < step; j++)
								{
									top_data[offset + j] =
										bottom_data[offset + j] >= 0.0f ? bottom_data[offset + j] : slope_data[0] * bottom_data[offset + j];
								}
							}
							else
							{
//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
								for (int j = 0; j < step; j++)
								{
									top_data[offset + j] =
										bottom_data[offset + j] >= 0.0f ? bottom_data[offset + j] : slope_data[c] * bottom_data[offset + j];
								}
							}
						}
					}
				}
				else if (bottoms[i]->order() == memory::NHWC)
				{
					for (size_t n = 0; n < bottoms[i]->num(); n++)
					{
						NOT_IMPLEMENTED;
					}
				}
				else
				{
					LOG(FATAL) << "Un-supported Order Type.";
				}
			}
		}


		template<typename Dtype>
		void operation_prelu<Dtype>::forward_gpu_f32(
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

		INSTANCE_CLASS(operation_prelu);
		REGISTE(operation_prelu);
	}
}