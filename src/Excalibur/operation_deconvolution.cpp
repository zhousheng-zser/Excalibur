#include "../../include/Excalibur/operation_deconvlution.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"
#include "../../include/Excalibur/math_functions.hpp"
#include "../../include/Excalibur/im2col.hpp"
#include <random>

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_deconvolution<Dtype>::operation_deconvolution(const operation_param& param) : operation_general_conv<Dtype>(param)
		{

		}

		template<typename Dtype>
		int operation_deconvolution<Dtype>::init_weights(FILE *fp)
		{
			int quantize_tag;
			fread(&quantize_tag, 1, sizeof(int), fp);
			int mem = 0;
			if (quantize_tag == 0)
			{
				weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(weight_data_size_, params_.device_, memory::NCHW, nullptr)));
				fread(weights_f32_[0]->mutable_cpu_data(), 1, weight_data_size_ * sizeof(float), fp);
				mem += weight_data_size_ * sizeof(float);
				if (bias_term_)
				{
					weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(output_channel_, params_.device_, memory::NCHW, nullptr)));
					fread(weights_f32_[1]->mutable_cpu_data(), 1, output_channel_ * sizeof(float), fp);
					mem += output_channel_ * sizeof(float);
				}
				return mem;
			}
			else
			{
				NOT_IMPLEMENTED;
				return 0;
			}
		}

		template<typename Dtype>
		int operation_deconvolution<Dtype>::init_weights()
		{
			std::default_random_engine e;
			std::normal_distribution<float> n(0, 0.3);
			std::uniform_int_distribution<int> u(-128, 127);
			int mem = 0;
			if (!params_.int8_quantization_)
			{
				weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(weight_data_size_, params_.device_, memory::NCHW, nullptr)));
				for (size_t i = 0; i < weight_data_size_; i++)
				{
					weights_f32_[0]->mutable_cpu_data()[i] = n(e);
				}
				mem += weight_data_size_ * sizeof(float);
				if (bias_term_)
				{
					weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(output_channel_, params_.device_, memory::NCHW, nullptr)));
					for (size_t i = 0; i < output_channel_; i++)
					{
						weights_f32_[1]->mutable_cpu_data()[i] = n(e);
					}
					mem += output_channel_ * sizeof(float);
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
			return mem;
		}

		template<typename Dtype>
		void operation_deconvolution<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			CHECK_EQ(tops.size(), 1);
			num_ = bottoms[0]->data_shape()[0];
			bottom_dim_ = bottoms[0]->count(1, 4);
			const float* bottom_data = bottoms[0]->cpu_data();
			const float* weights = weights_f32_[0]->cpu_data();
			const float* bias = nullptr;
			if (bottoms[0]->order() != memory::NCHW)
			{
				bottoms[0]->convert_order();
			}
			input_channel_ = bottoms[0]->data_shape()[1];
			input_dim_h_ = bottoms[0]->data_shape()[2];
			input_dim_w_ = bottoms[0]->data_shape()[3];
			input_spatial_dim_ = input_dim_h_ * input_dim_w_;
			if (output_dim_h_ == 0)
			{
				output_dim_h_ = (input_dim_h_ - 1) * stride_h_ + kernel_size_h_ - (pad_top_ + pad_bottom_);
			}
			if (output_dim_w_ == 0)
			{
				output_dim_w_ = (input_dim_w_ - 1) * stride_w_ + kernel_size_w_ - (pad_left_ + pad_right_);
			}
			output_spatial_dim_ = output_dim_w_ * output_dim_h_;
			tops[0].reset(new memory::tensor<float>(std::vector<int>{num_, output_channel_, output_dim_h_, output_dim_w_},
				bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
			float* top_data = tops[0]->mutable_cpu_data();
			top_dim_ = tops[0]->count(1, 4);
			col_buffer_.reset(new memory::tensor<float>(std::vector<int>{1, output_channel_ * kernel_size_h_ * kernel_size_w_, input_dim_h_, input_dim_w_},
				-1, memory::NCHW, nullptr));
			col_offset_ = input_spatial_dim_ * kernel_size_h_ * kernel_size_w_;
			output_offset_ = output_channel_ * output_spatial_dim_ / group_;
			if (bias_term_)
			{
				math_functions::cpu_set(output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
				bias = weights_f32_[1]->cpu_data();
			}
			for (int n = 0; n < num_; n++)
			{
				forward_sgemm(bottom_data + n * bottom_dim_, weights, top_data + n * top_dim_, bottoms[0]->order());
				if (bias_term_)
				{
					forward_sbias(top_data + n * top_dim_, bias, bottoms[0]->order());
				}
			}
		}

//		template<typename Dtype>
//		void operation_deconvolution<Dtype>::forward_gpu_f32(
//#ifdef USE_CUDA
//			cublasHandle_t &cublas_handle_,
//#ifdef USE_CUDNN
//			cudnnHandle_t cudnn_handle,
//#endif //!USE_CUDNN
//#endif //!USE_CUDA
//			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
//			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
//		{
//
//		}


		template<typename Dtype>
		void operation_deconvolution<Dtype>::forward_sgemm(const float* input, const float* weights, float* output, memory::orderType order)
		{
			col_buffer_data = col_buffer_->mutable_cpu_data();
			if (order == memory::NCHW)
			{
				for (int g = 0; g < group_; g++)
				{
					math_functions::cpu_sgemm(CblasTrans, CblasNoTrans, output_channel_ * kernel_size_h_ * kernel_size_w_ / group_,
						input_spatial_dim_, input_channel_ / group_, 1.0f,
						weights + kernel_size_h_ * kernel_size_w_ * g, 
						input + g * input_channel_ * input_spatial_dim_ / group_, 
						0.0f, col_buffer_data + col_offset_ * g);
				}
				col2im_cpu(col_buffer_data, input_channel_, output_dim_h_, output_dim_w_, kernel_size_h_,
					kernel_size_w_, pad_left_, pad_top_, stride_h_, stride_w_, dilation_h_, dilation_w_, output);
			}
			else if (order == memory::NHWC)
			{
				NOT_IMPLEMENTED;
				//if (group_ == 1)
				//{
				//	math_functions::cpu_sgemm(CblasNoTrans, CblasTrans, input_spatial_dim_, output_Channel_ * kernel_length_,
				//		input_Channel_, 1.0f,
				//		input, weights, 0.0f, col_buff);
				//}
				//else if (group_ > 1)
				//{
				//	float* temp = (float*)malloc(input_spatial_dim_ * kernel_length_ * sizeof(float));
				//	math_functions::cpu_sgemm(CblasNoTrans, CblasTrans, input_spatial_dim_, output_Channel_ * kernel_length_ / group_,
				//		input_Channel_, 1.0f,
				//		input, weights, 0.0f, temp);

				//	for (int i = 0; i < input_spatial_dim_; i++)
				//	{
				//		for (int j = 0; j < output_Channel_; j++)
				//		{
				//			memcpy(col_buff + i * output_Channel_ * kernel_length_ + j * kernel_length_, 
				//				temp + i * kernel_length_, kernel_length_ * sizeof(float));
				//		}
				//	}

				//	free(temp);
				//}
				//else
				//{
				//	LOG(FATAL) << "illegal group!";
				//}

				//conv_col2im_cpu(col_buff, output);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		template<typename Dtype>
		void operation_deconvolution<Dtype>::forward_sbias(float* output, const float* bias, memory::orderType order)
		{
			if (order == memory::NCHW)
			{
				math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_channel_,
					output_spatial_dim_, 1, 1.0f, bias, bias_multiplier_->cpu_data(),
					1.0f, output);
			}
			else if (order == memory::NHWC)
			{
				math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_spatial_dim_,
					output_channel_, 1, 1.0f, bias_multiplier_->cpu_data(), bias,
					1.0f, output);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		INSTANCE_CLASS(operation_deconvolution);
		REGISTE(operation_deconvolution);
	}
}