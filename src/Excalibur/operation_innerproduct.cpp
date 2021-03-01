#include "../../include/Excalibur/operation_innerproduct.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"
#include "../../include/Excalibur/math_functions.hpp"
#include "../../include/Excalibur/im2col.hpp"
#include <random>

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_innerproduct<Dtype>::operation_innerproduct(const operation_param& param) : operation<Dtype>(param)
		{
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					num_output_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "1")
				{
					this->bias_term_ = (bool)atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "2")
				{
					weight_data_size_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "8")
				{
					int8_scale_term_ = (bool)atoi(split_string(attrs[i], "=")[1].c_str());
					this->params_.set_int8_quantization(int8_scale_term_);
				}
				else if (split_string(attrs[i], "=")[0] == "9")
				{
					activation_type_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "10")
				{
					NOT_IMPLEMENTED;
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
		int operation_innerproduct<Dtype>::init_weights(FILE* fp)
		{
			int quantize_tag;
			fread(&quantize_tag, 1, sizeof(int), fp);
			int mem = 0;
			if (quantize_tag == 0)
			{
				this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(weight_data_size_, this->params_.device_, memory::NCHW, nullptr)));
				fread(this->weights_f32_[0]->mutable_cpu_data(), 1, weight_data_size_ * sizeof(float), fp);
				mem += weight_data_size_ * sizeof(float);
				if (this->bias_term_)
				{
					this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(num_output_, this->params_.device_, memory::NCHW, nullptr)));
					fread(this->weights_f32_[1]->mutable_cpu_data(), 1, num_output_ * sizeof(float), fp);
					mem += num_output_ * sizeof(float);
				}
				return mem;
			}
			else if (quantize_tag == 871224)
			{
				size_t align_data_size = (weight_data_size_ + 4 - 1) & -4;
				this->weights_i8_.push_back(std::shared_ptr<memory::tensor<signed char>>(new memory::tensor<signed char>(align_data_size, this->params_.device_, memory::NCHW, nullptr)));
				fread(this->weights_i8_[0]->mutable_cpu_data(), 1, align_data_size, fp);
				mem += align_data_size;
				if (this->bias_term_)
				{
					this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(1, this->params_.device_, memory::NCHW, nullptr)));
					this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(num_output_, this->params_.device_, memory::NCHW, nullptr)));
					fread(this->weights_f32_[1]->mutable_cpu_data(), 1, num_output_ * sizeof(float), fp);
					mem += num_output_ * sizeof(float);
				}
				this->weights_scaletable_i8_.resize(1);
				fread(this->weights_scaletable_i8_.data(), 1, 1 * sizeof(float), fp);
				this->featmap_scaletable_i8_.resize(1);
				fread(this->featmap_scaletable_i8_.data(), 1, 1 * sizeof(float), fp);
				mem += 2 * sizeof(float);
				return mem;
			}
			else
			{
				NOT_IMPLEMENTED;
				return 0;
			}
		}

		template<typename Dtype>
		int operation_innerproduct<Dtype>::init_weights()
		{
			std::default_random_engine e;
			std::normal_distribution<float> n(0, 0.3);
			std::uniform_int_distribution<int> u(-128, 127);
			int mem = 0;
			if (!this->params_.int8_quantization_)
			{
				this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(weight_data_size_, this->params_.device_, memory::NCHW, nullptr)));
				for (size_t i = 0; i < weight_data_size_; i++)
				{
					this->weights_f32_[0]->mutable_cpu_data()[i] = n(e);
				}
				mem += weight_data_size_ * sizeof(float);
				if (this->bias_term_)
				{
					this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(num_output_, this->params_.device_, memory::NCHW, nullptr)));
					for (size_t i = 0; i < num_output_; i++)
					{
						this->weights_f32_[1]->mutable_cpu_data()[i] = n(e);
					}
					mem += num_output_ * sizeof(float);
				}
			}
			else
			{
				size_t align_data_size = (weight_data_size_ + 4 - 1) & -4;
				this->weights_i8_.push_back(std::shared_ptr<memory::tensor<signed char>>(new memory::tensor<signed char>(align_data_size, this->params_.device_, memory::NCHW, nullptr)));
				for (size_t i = 0; i < align_data_size; i++)
				{
					this->weights_i8_[0]->mutable_cpu_data()[i] = u(e);
				}
				mem += align_data_size;
				if (this->bias_term_)
				{
					this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(1, this->params_.device_, memory::NCHW, nullptr)));
					this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(num_output_, this->params_.device_, memory::NCHW, nullptr)));
					for (size_t i = 0; i < num_output_; i++)
					{
						this->weights_f32_[1]->mutable_cpu_data()[i] = n(e);
					}
					mem += num_output_ * sizeof(float);
				}
				this->weights_scaletable_i8_.resize(1);
				this->weights_scaletable_i8_[0] = n(e);
				this->featmap_scaletable_i8_.resize(1);
				this->featmap_scaletable_i8_[0] = n(e);
				mem += 2 * sizeof(float);
			}
			return mem;
		}

		template<typename Dtype>
		void operation_innerproduct<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			CHECK_EQ(tops.size(), 1);
			int m = bottoms[0]->num();
			int n = num_output_;
			int k = bottoms[0]->count(1, 4);
			if (this->bias_term_)
			{
				bias_multiplier_.reset(new memory::tensor<float>(std::vector<int>{m}, -1, memory::NCHW, nullptr));
				math_functions::cpu_set(m, 1.0f, bias_multiplier_->mutable_cpu_data());
			}
			if (bottoms[0]->order() == memory::NCHW)
			{
				tops[0].reset(new memory::tensor<float>(std::vector<int>{m, n, 1, 1}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
				const float* bottom_data = bottoms[0]->cpu_data();
				float* top_data = tops[0]->mutable_cpu_data();
				const float* weight = this->weights_f32_[0]->cpu_data();

				math_functions::cpu_sgemm(CblasNoTrans, CblasTrans, m, n, k, 1.0f,
					bottom_data, weight, 0.0f, top_data);
				if (this->bias_term_)
				{
					math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, m, n, 1, 1.0f,
						bias_multiplier_->cpu_data(), this->weights_f32_[1]->cpu_data(), 1.0f, top_data);
				}
			}
			else if (bottoms[0]->order() == memory::NHWC)
			{
				tops[0].reset(new memory::tensor<float>(std::vector<int>{m, 1, 1, n}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
				const float* bottom_data = bottoms[0]->cpu_data();
				float* top_data = tops[0]->mutable_cpu_data();
				const float* weight = this->weights_f32_[0]->cpu_data();

				if (bottoms[0]->height() != 1 || bottoms[0]->width() != 1)
				{
					std::shared_ptr<memory::tensor<float>> col_buff;
					col_buff.reset(new memory::tensor<float>(std::vector<int>{bottoms[0]->num(), bottoms[0]->height(), bottoms[0]->width(), bottoms[0]->channels()},
						-1, memory::NCHW, nullptr));
					float* col_buff_data = col_buff->mutable_cpu_data();

					im2col_cpu(bottom_data, bottoms[0]->channels(), bottoms[0]->height(), bottoms[0]->width(), 1,
						1, 0, 0, 1, 1, 1, 1, col_buff_data, bottoms[0]->order(), bottoms[0]->num());
					math_functions::cpu_sgemm(CblasNoTrans, CblasTrans, m, n, k, 1.0f,
						col_buff_data, weight, 0.0f, top_data);
				}
				else
				{
					math_functions::cpu_sgemm(CblasNoTrans, CblasTrans, m, n, k, 1.0f,
						bottom_data, weight, 0.0f, top_data);
				}

				if (this->bias_term_)
				{
					math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, m, n, 1, 1.0f,
						bias_multiplier_->cpu_data(), this->weights_f32_[1]->cpu_data(), 1.0f, top_data);
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}


		template<typename Dtype>
		void operation_innerproduct<Dtype>::forward_gpu_f32(
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

		INSTANCE_CLASS(operation_innerproduct);
		REGISTE(operation_innerproduct);
	}
}