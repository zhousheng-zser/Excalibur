#include "../../include/Excalibur/operation_batchnorm.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"

#include <random>

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_batchnorm<Dtype>::operation_batchnorm(const operation_param& param) : operation<Dtype>(param)
		{
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					channels_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "1")
				{
					eps_ = atof(split_string(attrs[i], "=")[1].c_str());
				}
				else
				{
					LOG(FATAL) << "Un-supported Batchnorm Attribution " << split_string(attrs[i], "=")[0];
				}
			}
			this->params_.inplace_ = true;
		}

		template<typename Dtype>
		int operation_batchnorm<Dtype>::init_weights(FILE* fp)
		{
			// slope
			this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(channels_, this->params_.device_, memory::NCHW, nullptr)));
			fread(this->weights_f32_[0]->mutable_cpu_data(), 1, channels_ * sizeof(float), fp);
			auto slope_data = this->weights_f32_[0]->cpu_data();
			// mean
			this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(channels_, this->params_.device_, memory::NCHW, nullptr)));
			fread(this->weights_f32_[1]->mutable_cpu_data(), 1, channels_ * sizeof(float), fp);
			auto mean_data = this->weights_f32_[1]->cpu_data();
			// var
			this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(channels_, this->params_.device_, memory::NCHW, nullptr)));
			fread(this->weights_f32_[2]->mutable_cpu_data(), 1, channels_ * sizeof(float), fp);
			auto var_data = this->weights_f32_[2]->cpu_data();
			// bias
			this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(channels_, this->params_.device_, memory::NCHW, nullptr)));
			fread(this->weights_f32_[3]->mutable_cpu_data(), 1, channels_ * sizeof(float), fp);
			auto bias_data = this->weights_f32_[3]->cpu_data();

			//a = bias - slope * mean / sqrt(var)
			a_.reset(new memory::tensor<float>(channels_, this->params_.device_, memory::NCHW, nullptr));
			auto a_data = a_->mutable_cpu_data();
			//b = slope / sqrt(var)
			b_.reset(new memory::tensor<float>(channels_, this->params_.device_, memory::NCHW, nullptr));
			auto b_data = b_->mutable_cpu_data();

			for (size_t i = 0; i < channels_; i++)
			{
				float sqrt_var = static_cast<float>(sqrt(var_data[i] + eps_));
				a_data[i] = bias_data[i] - slope_data[i] * mean_data[i] / sqrt_var;
				b_data[i] = slope_data[i] / sqrt_var;
			}
			return channels_ * sizeof(float) * 4;
		}

		template<typename Dtype>
		int operation_batchnorm<Dtype>::init_weights()
		{
			std::default_random_engine e;
			std::normal_distribution<float> n(0, 0.3);
			std::uniform_int_distribution<int> u(-128, 127);

			// slope
			this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(channels_, this->params_.device_, memory::NCHW, nullptr)));
			for (size_t i = 0; i < channels_; i++)
			{
				this->weights_f32_[0]->mutable_cpu_data()[i] = n(e);
			}
			auto slope_data = this->weights_f32_[0]->cpu_data();
			// mean
			this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(channels_, this->params_.device_, memory::NCHW, nullptr)));
			for (size_t i = 0; i < channels_; i++)
			{
				this->weights_f32_[1]->mutable_cpu_data()[i] = n(e);
			}
			auto mean_data = this->weights_f32_[1]->cpu_data();
			// var
			this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(channels_, this->params_.device_, memory::NCHW, nullptr)));
			for (size_t i = 0; i < channels_; i++)
			{
				this->weights_f32_[2]->mutable_cpu_data()[i] = n(e);
			}
			auto var_data = this->weights_f32_[2]->cpu_data();
			// bias
			this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(channels_, this->params_.device_, memory::NCHW, nullptr)));
			for (size_t i = 0; i < channels_; i++)
			{
				this->weights_f32_[3]->mutable_cpu_data()[i] = n(e);
			}
			auto bias_data = this->weights_f32_[3]->cpu_data();

			//a = bias - slope * mean / sqrt(var)
			a_.reset(new memory::tensor<float>(channels_, this->params_.device_, memory::NCHW, nullptr));
			auto a_data = a_->mutable_cpu_data();
			//b = slope / sqrt(var)
			b_.reset(new memory::tensor<float>(channels_, this->params_.device_, memory::NCHW, nullptr));
			auto b_data = b_->mutable_cpu_data();

			for (size_t i = 0; i < channels_; i++)
			{
				float sqrt_var = static_cast<float>(sqrt(var_data[i] + eps_));
				a_data[i] = bias_data[i] - slope_data[i] * mean_data[i] / sqrt_var;
				b_data[i] = slope_data[i] / sqrt_var;
			}
			return channels_ * sizeof(float) * 4;
		}


		template<typename Dtype>
		void operation_batchnorm<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), tops.size());
			for (size_t i = 0; i < bottoms.size(); i++)
			{
				tops[i].reset(new memory::tensor<float>(bottoms[i]->data_shape(), bottoms[i]->device(), bottoms[i]->order(), bottoms[i]->allocator()));
				auto bottom_data = bottoms[i]->cpu_data();
				auto top_data = tops[i]->mutable_cpu_data();
				auto a_data = a_->cpu_data();
				auto b_data = b_->cpu_data();
				auto offset_n = bottoms[i]->count(1, 4);
				if (bottoms[i]->order() == memory::NCHW)
				{
					for (size_t n = 0; n < bottoms[i]->num(); n++)
					{
						auto offset_c = bottoms[i]->count(2, 4);
						for (size_t c = 0; c < bottoms[i]->channels(); c++)
						{
#ifdef _OPENMP
#pragma omp parallel for
#endif // !_OPENMP
							for (int j = 0; j < offset_c; j++)
							{
								// value = b * value + a
								top_data[offset_n * n + offset_c * c + j] = b_data[c] * bottom_data[offset_n * n + offset_c * c + j] + a_data[c];
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


		template<typename Dtype>
		void operation_batchnorm<Dtype>::forward_gpu_f32(
#ifdef USE_CUDA
			cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			NOT_IMPLEMENTED;
		}

		INSTANCE_CLASS(operation_batchnorm);
		REGISTE(operation_batchnorm);
	}
}