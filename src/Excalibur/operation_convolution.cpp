#include "../../include/Excalibur/operation_convolution.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"
#include "../../include/Excalibur/im2col.hpp"
#include "../../include/Excalibur/math_functions.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_convolution<Dtype>::operation_convolution(const operation_param& param) : operation_general_conv<Dtype>(param)
		{
			
		}

		template<typename Dtype>
		int operation_convolution<Dtype>::init_weights(FILE *fp)
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
			}
			else if (quantize_tag == 871224)
			{
				weights_i8_.push_back(std::shared_ptr<memory::tensor<signed char>>(new memory::tensor<signed char>(weight_data_size_, params_.device_, memory::NCHW, nullptr)));
				fread(weights_i8_[0]->mutable_cpu_data(), 1, weight_data_size_ * sizeof(signed char), fp);
				// fake float32 data, just for code consistency
				weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(1, params_.device_, memory::NCHW, nullptr)));
				mem += weight_data_size_ * sizeof(signed char);
				if (weight_data_size_ % 4 != 0)
				{
					fread(weights_f32_[0]->mutable_cpu_data(), 1, (4 - weight_data_size_ % 4) * sizeof(signed char), fp);
					mem += 1 * sizeof(float);
				}
				if (bias_term_)
				{
					weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(output_channel_, params_.device_, memory::NCHW, nullptr)));
					fread(weights_f32_[1]->mutable_cpu_data(), 1, output_channel_ * sizeof(float), fp);
					mem += output_channel_ * sizeof(signed char);
				}
				weights_scaletable_i8_.resize(group_);
				fread(weights_scaletable_i8_.data(), 1, group_ * sizeof(float), fp);
				featmap_scaletable_i8_.resize(1);
				fread(featmap_scaletable_i8_.data(), 1, 1 * sizeof(float), fp);
				mem += (group_ + 1) * sizeof(signed char);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
			return mem;
		}

		template<typename Dtype>
		void operation_convolution<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			CHECK_EQ(tops.size(), 1);
			memory::orderType order = bottoms[0]->order();
			num_ = bottoms[0]->num();
			const float* bottom_data = bottoms[0]->cpu_data();
			const float* weights_data = weights_f32_[0]->cpu_data();
			const float* bias_data = nullptr;
			input_channel_ = bottoms[0]->channels();
			input_dim_h_ = bottoms[0]->height();
			input_dim_w_ = bottoms[0]->width();
			output_dim_h_ = (input_dim_h_ + pad_bottom_ + pad_top_ - kernel_size_h_) / stride_h_ + 1;
			output_dim_w_ = (input_dim_w_ + pad_left_ + pad_right_ - kernel_size_w_) / stride_w_ + 1;
			output_spatial_dim_ = output_dim_w_ * output_dim_h_;
			kernel_dim_ = input_channel_ * kernel_size_h_* kernel_size_w_;
			col_offset_ = kernel_dim_ * output_spatial_dim_;
			output_offset_ = output_channel_ * output_spatial_dim_ / group_;
			if (order == memory::NCHW)
			{
				tops[0].reset(new memory::tensor<float>(std::vector<int>{num_, output_channel_, output_dim_h_, output_dim_w_}, params_.device_, order, nullptr));
			}
			else if (order == memory::NHWC)
			{
				tops[0].reset(new memory::tensor<float>(std::vector<int>{num_, output_dim_h_, output_dim_w_, output_channel_}, params_.device_, order, nullptr));
			}
			else
			{
				LOG(FATAL) << "Un-supported data arrange.";
			}
			float* top_data = tops[0]->mutable_cpu_data();
			int top_dim_ = tops[0]->count(1, 4);
			col_buffer_.reset(new memory::tensor<float>(std::vector<int>{kernel_dim_ / group_, output_dim_h_, output_dim_w_}, params_.device_, memory::NCHW, nullptr));
			col_buffer_data = col_buffer_->mutable_cpu_data();
			if (bias_term_)
			{
				bias_multiplier_.reset(new memory::tensor<float>(std::vector<int>{output_dim_w_*output_dim_h_}, params_.device_, memory::NCHW, nullptr));
				bias_multiplier_data = bias_multiplier_->mutable_cpu_data();
				math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_data);
				bias_data = weights_f32_[1]->cpu_data();
			}
			if ((kernel_size_h_ == 1 && kernel_size_w_ == 1) && (stride_h_ == 1 && stride_w_ == 1))
			{
				CHECK_EQ(pad_left_ + pad_right_ + pad_top_ + pad_bottom_, 0);
				forward_cpu_k1s1_f32(bottoms[0], tops[0]);
			}
			else
			{
				for (int n = 0; n < num_; n++)
				{
					forward_cpu_sgemm(bottom_data + n * bottom_dim_, weights_data, top_data + n * top_dim_, order);
					if (bias_term_)
					{
						forward_cpu_sbias(top_data + n * top_dim_, bias_data, order);
					}
				}
			}			
		}


		template<typename Dtype>
		void operation_convolution<Dtype>::forward_cpu_sgemm(const float* input, const float* weights, float* output, memory::orderType order)
		{
			const float* col_buff = input;
			//if ((kernel_size_h_ * kernel_size_w_ != 1) || (order == memory::NHWC))
			//{
			//	//conv_im2col_cpu(input, col_buffer_data);
			//	im2col_cpu(input, input_channel_, input_dim_h_, input_dim_w_, kernel_size_h_,
			//		kernel_size_w_, pad_left_, pad_top_, stride_h_, stride_w_, dilation_h_, dilation_w_, col_buffer_data, order);
			//	col_buff = col_buffer_data;
			//	
			//}

			if (order == memory::NCHW)
			{
				if (group_ == 1)
				{
					im2col_cpu(input, input_channel_, input_dim_h_, input_dim_w_, kernel_size_h_,
					kernel_size_w_, pad_left_, pad_top_, stride_h_, stride_w_, dilation_h_, dilation_w_, col_buffer_data, order);
					col_buff = col_buffer_data;
					math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_channel_,
						output_spatial_dim_, kernel_dim_, 1.0f,
						weights, col_buff, 0.0f, output);
				}
				else
				{
					for (int g = 0; g < group_; ++g)
					{
						int gistep = input_channel_ / group_;
						int gostep = output_channel_ / group_;
						im2col_cpu(input + input_dim_h_* input_dim_w_ * gistep * g, gistep, input_dim_h_, input_dim_w_, kernel_size_h_,
							kernel_size_w_, pad_left_, pad_top_, stride_h_, stride_w_, dilation_h_, dilation_w_, col_buffer_data, order);
						col_buff = col_buffer_data;
						math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, gostep,
							output_spatial_dim_, kernel_size_h_ * kernel_size_w_ * gistep, 1.0f,
							weights + kernel_size_h_ * kernel_size_w_ * g * gostep,
							col_buff, 0.0f, output + output_spatial_dim_ * g * gostep);
					}
				}
			}
			else if (order == memory::NHWC)
			{
				if (group_ == 1)
				{
					math_functions::cpu_sgemm(CblasTrans, CblasTrans, output_spatial_dim_, output_channel_,
						kernel_dim_, 1.0f,
						col_buff, weights, 0.0f, output);
				}
				else
				{
					for (int g = 0; g < group_; ++g)
					{
						math_functions::cpu_sgemm(CblasTrans, CblasTrans, output_spatial_dim_, output_channel_ / group_,
							kernel_size_h_ * kernel_size_w_, 1.0f,
							col_buff + output_spatial_dim_ * kernel_size_h_ * kernel_size_w_ * g,
							weights + kernel_size_h_ * kernel_size_w_ * g, 0.0f,
							output + output_spatial_dim_ * g);
					}

					std::shared_ptr<memory::tensor<float>> temp;
					temp.reset(new memory::tensor<float>(std::vector<int>{output_spatial_dim_ * output_channel_}));
					float* temp_data = temp->mutable_cpu_data();

					for (int ch = 0; ch < output_channel_; ++ch)
					{
						int channel_offset = ch * output_dim_h_ * output_dim_w_;
						for (int row = 0; row < output_dim_h_; ++row)
						{
							int row_offset = row * output_dim_w_;
							for (int col = 0; col < output_dim_w_; ++col)
							{
								temp_data[(row_offset + col) * output_channel_ + ch] = output[channel_offset + row_offset + col];
							}
						}
					}
					memcpy(output, temp_data, output_spatial_dim_ * output_channel_ * sizeof(float));
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		template<typename Dtype>
		void operation_convolution<Dtype>::forward_cpu_sbias(float* output, const float* bias, memory::orderType order)
		{
			if (order == memory::NCHW)
			{
				math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_channel_,
					output_spatial_dim_, 1, 1.0f, bias, bias_multiplier_data,
					1.0f, output);
			}
			else if (order == memory::NHWC)
			{
				math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_spatial_dim_,
					output_channel_, 1, 1.0f, bias_multiplier_data, bias,
					1.0f, output);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}


		template<typename Dtype>
		void operation_convolution<Dtype>::forward_cpu_k1s1_f32(const std::shared_ptr < memory::tensor<float>>& bottom,
			std::shared_ptr < memory::tensor<float>>& top)
		{
			float* top_data = top->mutable_cpu_data();
			const float* bottom_data = bottom->cpu_data();
			const float* weights_data = weights_f32_[0]->cpu_data();
			const int step = bottom->count(2, 4);
			memset(top_data, 0, top->count() * sizeof(float));
//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
			for (int i = 0; i < output_channel_; i++)
			{
				float* top_data_slice = top_data + i * step;
				for (int j = 0; j < input_channel_; j++)
				{
					cblas_saxpy(step, weights_data[i * input_channel_ + j], bottom_data + j * step, 1, top_data_slice, 1);
				}
			}
			if (bias_term_)
			{
				forward_cpu_sbias(top_data, weights_f32_[1]->cpu_data(), memory::NCHW);
			}
		}

#ifndef USE_CUDA
		STUB_GPU(operation_convolution);
#endif // !USE_CUDA


		INSTANCE_CLASS(operation_convolution);
		REGISTE(operation_convolution);
	}
}