#include "../../include/Excalibur/operation_memorydata.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"
#include "../../include/Excalibur/math_functions.hpp"
#include <random>

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_memorydata<Dtype>::operation_memorydata(const operation_param& param) : operation<Dtype>(param)
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
					h_ = (bool)atoi(split_string(attrs[i], "=")[1].c_str());
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
					LOG(FATAL) << "Un-supported InnerProduct Attribution " << split_string(attrs[i], "=")[0];
				}
			}
		}

		template<typename Dtype>
		int operation_memorydata<Dtype>::init_weights(FILE* fp)
		{
			int mem = 0;
			if (c_ != 0)
			{
				this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(c_, h_ , w_, this->params_.device_, memory::NCHW, nullptr)));
				fread(this->weights_f32_[0]->mutable_cpu_data(), 1, c_ * h_ * w_ * sizeof(float), fp);
				mem += c_ * h_ * w_ * sizeof(float);
			}
			else if (h_ != 0)
			{
				this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(h_, w_, this->params_.device_, memory::NCHW, nullptr)));
				fread(this->weights_f32_[0]->mutable_cpu_data(), 1, h_ * w_ * sizeof(float), fp);
				mem += h_ * w_ * sizeof(float);
			}
			else if (w_ != 0)
			{
				this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(w_, this->params_.device_, memory::NCHW, nullptr)));
				fread(this->weights_f32_[0]->mutable_cpu_data(), 1, w_ * sizeof(float), fp);
				mem += w_ * sizeof(float);
			}
			
			return mem;
		}

		template<typename Dtype>
		int operation_memorydata<Dtype>::init_weights()
		{
			std::default_random_engine e;
			std::normal_distribution<float> n(0, 0.3);
			std::uniform_int_distribution<int> u(-128, 127);
			int mem = 0;
			if (c_ != 0)
			{
				this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(c_, h_, w_, this->params_.device_, memory::NCHW, nullptr)));
				for (size_t i = 0; i < c_ * h_ * w_; i++)
				{
					this->weights_f32_[0]->mutable_cpu_data()[i] = n(e);
				}
				mem += c_ * h_ * w_ * sizeof(float);
			}
			else if (h_ != 0)
			{
				this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(h_, w_, this->params_.device_, memory::NCHW, nullptr)));
				for (size_t i = 0; i < h_ * w_; i++)
				{
					this->weights_f32_[0]->mutable_cpu_data()[i] = n(e);
				}
				mem += h_ * w_ * sizeof(float);
			}
			else if (w_ != 0)
			{
				this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(w_, this->params_.device_, memory::NCHW, nullptr)));
				for (size_t i = 0; i < w_; i++)
				{
					this->weights_f32_[0]->mutable_cpu_data()[i] = n(e);
				}
				mem += w_ * sizeof(float);
			}

			return mem;
		}

		template<typename Dtype>
		void operation_memorydata<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(tops.size(), 1);

			tops[0] = this->weights_f32_[0];
		}

		template<typename Dtype>
		void operation_memorydata<Dtype>::forward_gpu_f32(
#ifdef USE_CUDA
			cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			forward_cpu_f32(bottoms, tops);
		}

		INSTANCE_CLASS(operation_memorydata);
		REGISTE(operation_memorydata);
	}
}