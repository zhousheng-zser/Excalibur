#include "../../include/Excalibur/operation_scale.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_scale<Dtype>::operation_scale(const operation_param& param) : operation<Dtype>(param)
		{
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					scale_data_size_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "1")
				{
					bias_term_ = (bool)atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "-23330")
				{
					//do nothing
				}
				else
				{
					LOG(FATAL) << "Un-supported Scale Attribution " << split_string(attrs[i], "=")[0];
				}
			}
			params_.inplace_ = true;
		}

		template<typename Dtype>
		int operation_scale<Dtype>::init_weights(FILE *fp)
		{
			int mem = 0;
			weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(scale_data_size_, params_.device_, memory::NCHW, nullptr)));
			fread(weights_f32_[0]->mutable_cpu_data(), 1, scale_data_size_ * sizeof(float), fp);
			mem += scale_data_size_;
			if (bias_term_)
			{
				weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(scale_data_size_, params_.device_, memory::NCHW, nullptr)));
				fread(weights_f32_[1]->mutable_cpu_data(), 1, scale_data_size_ * sizeof(float), fp);
				mem += scale_data_size_;
			}
			return mem * sizeof(float);
		}

		template<typename Dtype>
		void operation_scale<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), tops.size());
			for (size_t i = 0; i < bottoms.size(); i++)
			{
				CHECK_EQ(bottoms[i]->channels(), scale_data_size_);
				tops[i].reset(new memory::tensor<float>(bottoms[i]->data_shape(), bottoms[i]->device(), bottoms[i]->order(), bottoms[i]->allocator()));
				float* top_data = tops[i]->mutable_cpu_data();
				const float* bottom_data = bottoms[i]->cpu_data();
				const float* scale_data = weights_f32_[0]->cpu_data();
				const float* bias_data = nullptr;
				if (bias_term_)
				{
					bias_data = weights_f32_[1]->cpu_data();
				}
				if (bottoms[i]->order() == memory::NCHW)
				{
					for (size_t n = 0; n < bottoms[i]->num(); n++)
					{
						int ch = bottoms[i]->channels();
						for (size_t c = 0; c < ch; c++)
						{
							int step = bottoms[i]->count(2, 4);
							if (bias_term_)
							{
#ifdef _OPENMP
#pragma omp parallel for
#endif // !_OPENMP
								for (int j = 0; j < step; j++)
								{
									top_data[n * ch * step + c * step + j] =
										bottom_data[n * ch * step + c * step + j] * scale_data[c] + bias_data[c];
								}
							}
							else
							{
#ifdef _OPENMP
#pragma omp parallel for
#endif // !_OPENMP
								for (int j = 0; j < step; j++)
								{
									top_data[n * ch * step + c * step + j] =
										bottom_data[n * ch * step + c * step + j] * scale_data[c];
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


//		template<typename Dtype>
//		void operation_scale<Dtype>::forward_gpu_f32(
//#ifdef USE_CUDA
//			cublasHandle_t &cublas_handle_,
//#ifdef USE_CUDNN
//			cudnnHandle_t cudnn_handle,
//#endif //!USE_CUDNN
//#endif //!USE_CUDA
//			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
//			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
//		{
//			NOT_IMPLEMENTED;
//		}

		INSTANCE_CLASS(operation_scale);
		REGISTE(operation_scale);
	}
}