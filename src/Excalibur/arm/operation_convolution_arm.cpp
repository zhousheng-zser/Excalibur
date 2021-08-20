#include "../../../include/Excalibur/arm/operation_convolution_arm.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"
//#include "../../include/Excalibur/im2col.hpp"
#include "../../include/Excalibur/math_functions.hpp"
#include "../../include/Excalibur/operation_make_border.hpp"
#include "../../include/Excalibur/operation_cut_border.hpp"
//#include "../../include/Primitives/simd_types.hpp"
#include <random>

namespace glasssix
{
	namespace excalibur
	{
		static inline void fill(float *ptr, int size, float _v)
		{
#if __ARM_NEON
			int nn = size >> 2;
			int remain = size - (nn << 2);
#else
			int remain = size;

#endif // __ARM_NEON

#if __ARM_NEON
			float32x4_t _c = vdupq_n_f32(_v);
#if __aarch64__
			if (nn > 0)
			{
				asm volatile (
					"0:                             \n"
					"subs       %w0, %w0, #1        \n"
					"st1        {%4.4s}, [%1], #16  \n"
					"bne        0b                  \n"
					: "=r"(nn),     // %0
					"=r"(ptr)     // %1
					: "0"(nn),
					"1"(ptr),
					"w"(_c)       // %4
					: "cc", "memory"
					);
			}
#else
			if (nn > 0)
			{
				asm volatile(
					"0:                             \n"
					"subs       %0, #1              \n"
					"vst1.f32   {%e4-%f4}, [%1 :128]!\n"
					"bne        0b                  \n"
					: "=r"(nn),     // %0
					"=r"(ptr)     // %1
					: "0"(nn),
					"1"(ptr),
					"w"(_c)       // %4
					: "cc", "memory"
					);
			}
#endif // __aarch64__
#endif // __ARM_NEON
			for (; remain > 0; remain--)
			{
				*ptr++ = _v;
			}
		}

		static inline void fill(int *ptr, int size, int _v)
		{
#if __ARM_NEON
			int nn = size >> 2;
			int remain = size - (nn << 2);
#else
			int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
			int32x4_t _c = vdupq_n_s32(_v);
#if __aarch64__
			if (nn > 0)
			{
				asm volatile(
					"0:                             \n"
					"subs       %w0, %w0, #1        \n"
					"st1        {%4.4s}, [%1], #16  \n"
					"bne        0b                  \n"
					: "=r"(nn), // %0
					"=r"(ptr) // %1
					: "0"(nn),
					"1"(ptr),
					"w"(_c) // %4
					: "cc", "memory");
			}
#else
			if (nn > 0)
			{
				asm volatile(
					"0:                             \n"
					"subs       %0, #1              \n"
					"vst1.s32   {%e4-%f4}, [%1 :128]!\n"
					"bne        0b                  \n"
					: "=r"(nn), // %0
					"=r"(ptr) // %1
					: "0"(nn),
					"1"(ptr),
					"w"(_c) // %4
					: "cc", "memory");
			}
#endif // __aarch64__
#endif // __ARM_NEON
			for (; remain > 0; remain--)
			{
				*ptr++ = _v;
			}
		}

#include "../../../include/Excalibur/arm/convolution_sgemm.hpp"
#include "../../../include/Excalibur/arm/convolution_1x1.hpp"
#include "../../../include/Excalibur/arm/convolution_3x3.hpp"

#ifdef __ARM_NEON
#include "../../../include/Excalibur/arm/convolution_sgemm_pack4.hpp"
#endif


		template<typename Dtype>
		operation_convolution_arm<Dtype>::operation_convolution_arm(const operation_param& param) : operation_general_conv<Dtype>(param)
		{

		}

		template<typename Dtype>
		int operation_convolution_arm<Dtype>::init_weights(FILE* fp)
		{
			this->input_channel_ = this->weight_data_size_ / this->group_ / (this->output_channel_ / this->group_) / (this->kernel_size_w_ * this->kernel_size_h_) *  this->group_;
			int quantize_tag;
			fread(&quantize_tag, 1, sizeof(int), fp);
			int mem = 0;
			if (quantize_tag == 0)
			{
				this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(this->weight_data_size_, this->params_.device_, memory::NCHW, nullptr)));
				fread(this->weights_f32_[0]->mutable_cpu_data(), 1, this->weight_data_size_ * sizeof(float), fp);
				mem += this->weight_data_size_ * sizeof(float);
				if (this->bias_term_)
				{
					this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(this->output_channel_, this->params_.device_, memory::NCHW, nullptr)));
					fread(this->weights_f32_[1]->mutable_cpu_data(), 1, this->output_channel_ * sizeof(float), fp);
					mem += this->output_channel_ * sizeof(float);
				}

				if (this->group_ == 1)
				{
					const int maxk = this->kernel_size_h_ * this->kernel_size_w_;
					int elempack = this->input_channel_ % 4 ? 1 : 4;
					int out_elempack = this->output_channel_ % 4 ? 1 : 4;
#if __ARM_NEON
					if (elempack == 4 && out_elempack == 4)
					{
						bool prefer_sgemm = (this->dilation_w_ == 1 && this->dilation_h_ == 1 && this->stride_w_ == 1 && this->stride_h_ == 1
							&& this->input_channel_ >= 12 && this->output_channel_ >= 12)
							|| (this->dilation_w_ == 1 && this->dilation_h_ == 1 && this->stride_w_ == 2 && this->stride_h_ == 2
								&& this->input_channel_ >= 16 && this->output_channel_ >= 16)
							|| ((this->dilation_w_ >= 2 || this->dilation_h_ >= 2) && this->input_channel_ >= 16 && this->output_channel_ >= 16);

						if (this->kernel_size_w_ == 1 && this->kernel_size_h_ == 1 && this->dilation_w_ == 1 && this->dilation_h_ == 1 && this->stride_w_ == 1 && this->stride_h_ == 1)
						{
							convolution_im2col_sgemm_transform_kernel_pack4_neon(this->weights_f32_[0], weight_data_pack4_, this->input_channel_, this->output_channel_, this->kernel_size_w_, this->kernel_size_h_);
						}
						else if (this->kernel_size_w_ == 1 && this->kernel_size_h_ == 1 && this->dilation_w_ == 1 && this->dilation_h_ == 1 && this->stride_w_ == 2 && this->stride_h_ == 2)
						{
							convolution_im2col_sgemm_transform_kernel_pack4_neon(this->weights_f32_[0], weight_data_pack4_, this->input_channel_, this->output_channel_, this->kernel_size_w_, this->kernel_size_h_);
						}
						else if (this->kernel_size_w_ == 3 && this->kernel_size_h_ == 3 && this->dilation_w_ == 1 && this->dilation_h_ == 1 && this->stride_w_ == 1 && this->stride_h_ == 1)
						{
							conv3x3s1_winograd64_transform_kernel_pack4_neon(weight_data, weight_data_pack4, num_input, num_output);
							conv3x3s1_winograd42_transform_kernel_pack4_neon(weight_data, weight_3x3_winograd42_data_pack4, num_input, num_output);
						}
						else if (this->kernel_size_w_ == 3 && this->kernel_size_h_ == 3 && this->dilation_w_ == 1 && this->dilation_h_ == 1 && this->stride_w_ == 2 && this->stride_h_ == 2)
						{
							// we need more proper conditions
							if (opt.use_sgemm_convolution && this->input_channel_ >= 24 && this->output_channel_ >= 24)
							{
								convolution_im2col_sgemm_transform_kernel_pack4_neon(this->weights_f32_[0], weight_sgemm_data_pack4_, this->input_channel_, this->output_channel_, this->kernel_size_w_, this->kernel_size_h_);
							}

							convolution_transform_kernel_pack4_neon(weight_data, weight_data_pack4, num_input, num_output, kernel_w, this->kernel_size_h_);
						}
						else if (this->kernel_size_w_ == 5 && this->kernel_size_h_ == 5 && this->dilation_w_ == 1 && this->dilation_h_ == 1 && this->stride_w_ == 1 && this->stride_h_ == 1)
						{
							// we need more proper conditions
							if (opt.use_sgemm_convolution && this->input_channel_ >= 48 && thi->output_channel_ >= 48)
							{
								convolution_im2col_sgemm_transform_kernel_pack4_neon(this->weights_f32_[0], weight_sgemm_data_pack4_, this->input_channel_, this->output_channel_, this->kernel_size_w_, this->kernel_size_h_);
							}

							convolution_transform_kernel_pack4_neon(weight_data, weight_data_pack4, num_input, num_output, this->kernel_size_w_, this->kernel_size_h_);
						}
						else if (this->kernel_size_w_ == 5 && this->kernel_size_h_ == 5 && this->dilation_w_ == 1 && this->dilation_h_ == 1 && this->stride_w_ == 2 && this->stride_h_ == 2)
						{
							// we need more proper conditions
							if (opt.use_sgemm_convolution && this->input_channel_ >= 72 && thi->output_channel_ >= 72)
							{
								convolution_im2col_sgemm_transform_kernel_pack4_neon(this->weights_f32_[0], weight_sgemm_data_pack4_, this->input_channel_, this->output_channel_, this->kernel_size_w_, this->kernel_size_h_);
							}

							convolution_transform_kernel_pack4_neon(weight_data, weight_data_pack4, num_input, num_output, this->kernel_size_w_, this->kernel_size_h_);
						}
						else if (opt.use_sgemm_convolution && prefer_sgemm)
						{
							convolution_im2col_sgemm_transform_kernel_pack4_neon(this->weights_f32_[0], weight_sgemm_data_pack4_, this->input_channel_, this->output_channel_, this->kernel_size_w_, this->kernel_size_h_);
						}
						else
						{
							convolution_transform_kernel_pack4_neon(weight_data, weight_data_pack4, num_input, num_output, this->kernel_size_w_, this->kernel_size_h_);
						}
					}

					// pack1to4
					if (elempack == 1 && out_elempack == 4)
					{
						convolution_transform_kernel_pack1to4_neon(weight_data, weight_data_pack1to4, num_input, num_output, this->kernel_size_w_, this->kernel_size_h_);
					}

					// pack4to1
					if (elempack == 4 && out_elempack == 1)
					{
						if (this->kernel_size_w_ == 1 && this->kernel_size_h_ == 1 && this->this->dilation_w_ == 1 && this->this->dilation_h_ == 1 && this->this->stride_w_ == 1 && this->this->stride_h_ == 1)
						{
							conv1x1s1_sgemm_transform_kernel_pack4to1_neon(weight_data, weight_data_pack4to1, num_input, num_output);
						}
						else if (this->kernel_size_w_ == 1 && this->kernel_size_h_ == 1 && this->dilation_w_ == 1 && this->dilation_h_ == 1 && this->stride_w_ == 2 && this->stride_h_ == 2)
						{
							conv1x1s1_sgemm_transform_kernel_pack4to1_neon(weight_data, weight_data_pack4to1, num_input, num_output);
						}
						else if (this->kernel_size_w_ == 3 && this->kernel_size_h_ == 3 && this->dilation_w_ == 1 && this->dilation_h_ == 1 && this->stride_w_ == 1 && this->stride_h_ == 1)
						{
							conv3x3s1_winograd64_transform_kernel_pack4to1_neon(weight_data, weight_data_pack4to1, num_input, num_output);
						}
						else
						{
							convolution_transform_kernel_pack4to1_neon(weight_data, weight_data_pack4to1, num_input, num_output, this->kernel_size_w_, this->kernel_size_h_);
						}
					}
#endif
					if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3)
						&& (this->stride_h_ == 1 && this->stride_w_ == 1) 
						&& (this->dilation_h_ == 1 && this->dilation_w_ == 1)
						&& (this->input_channel_ / this->group_  >=16 && this->output_channel_ / this->group_ >=16))
					{
						if (this->input_channel_ >= 16 && this->output_channel_ >= 16)
							conv3x3s1_winograd64_transform_kernel_neon5(this->weights_f32_[0], kernel_tm_, this->input_channel_, this->output_channel_, this->group_);
					}
					else if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 2 && this->stride_w_ == 2) && (this->dilation_h_ == 1 && this->dilation_w_ == 1))
					{
						conv3x3s2_transform_kernel_neon(this->weights_f32_[0], kernel_tm_, this->input_channel_, this->output_channel_);
						convolution_im2col_sgemm_transform_kernel_neon(this->weights_f32_[0], kernel_tm_gemm_, this->input_channel_, this->output_channel_, this->kernel_size_w_, this->kernel_size_h_);
					}
					else if ((this->kernel_size_h_ == 1 && this->kernel_size_w_ == 1) && (this->stride_h_ == 1 && this->stride_w_ == 1) && (this->dilation_h_ == 1 && this->dilation_w_ == 1))
					{
						if(this->input_channel_ >= 64 && this->output_channel_ >= 64)
							convolution_im2col_sgemm_transform_kernel_neon(this->weights_f32_[0], kernel_tm_gemm_, this->input_channel_, this->output_channel_, this->kernel_size_w_, this->kernel_size_h_);
					}
					else
					{
						convolution_im2col_sgemm_transform_kernel_neon(this->weights_f32_[0], kernel_tm_gemm_, this->input_channel_, this->output_channel_, this->kernel_size_w_, this->kernel_size_h_);
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}
			else if(quantize_tag == 871224)
			{
				if (this->group_ == 1)
				{
					size_t align_data_size = (this->weight_data_size_ + 4 - 1) & -4;
					this->weights_i8_.push_back(std::shared_ptr<memory::tensor<signed char>>(new memory::tensor<signed char>(align_data_size, this->params_.device_, memory::NCHW, nullptr)));
					fread(this->weights_i8_[0]->mutable_cpu_data(), 1, align_data_size, fp);
					mem += align_data_size;
					if (this->bias_term_)
					{
						this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(1, this->params_.device_, memory::NCHW, nullptr)));
						this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(this->output_channel_, this->params_.device_, memory::NCHW, nullptr)));
						fread(this->weights_f32_[1]->mutable_cpu_data(), 1, this->output_channel_ * sizeof(float), fp);
						mem += this->output_channel_ * sizeof(float);
					}
					this->weights_scaletable_i8_.resize(this->output_channel_);
					fread(this->weights_scaletable_i8_.data(), 1, this->output_channel_ * sizeof(float), fp);
					this->featmap_scaletable_i8_.resize(1);
					fread(this->featmap_scaletable_i8_.data(), 1, 1 * sizeof(float), fp);
					mem += (this->output_channel_ + 1) * sizeof(float);
					if (this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3 && this->stride_h_ == 1 && this->stride_w_ == 1 && this->dilation_h_ == 1 && this->dilation_w_ == 1)
					{
						conv3x3s1_winograd43_transform_kernel_int8_neon();
					}
					else
					{
						conv_im2col_sgemm_transform_kernel_int8_neon();
					}
				}
				else
					NOT_IMPLEMENTED;
			}
			else
			{
				NOT_IMPLEMENTED;
			}
			return mem;
		}

		template<typename Dtype>
		int operation_convolution_arm<Dtype>::init_weights()
		{
			this->input_channel_ = this->weight_data_size_ / this->group_ / (this->output_channel_ / this->group_) / (this->kernel_size_w_ * this->kernel_size_h_) *  this->group_;

			std::default_random_engine e;
			std::normal_distribution<float> n(0, 0.3);
			std::uniform_int_distribution<int> u(-128, 127);
			int mem = 0;
			if (!this->params_.int8_quantization_)
			{
				this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(this->weight_data_size_, this->params_.device_, memory::NCHW, nullptr)));
				for (size_t i = 0; i < this->weight_data_size_; i++)
				{
					this->weights_f32_[0]->mutable_cpu_data()[i] = n(e);
				}
				mem += this->weight_data_size_ * sizeof(float);
				if (this->bias_term_)
				{
					this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(this->output_channel_, this->params_.device_, memory::NCHW, nullptr)));
					for (size_t i = 0; i < this->output_channel_; i++)
					{
						this->weights_f32_[1]->mutable_cpu_data()[i] = n(e);
					}
					mem += this->output_channel_ * sizeof(float);
				}

				if (this->group_ == 1)
				{
					if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3)
						&& (this->stride_h_ == 1 && this->stride_w_ == 1)
						&& (this->dilation_h_ == 1 && this->dilation_w_ == 1)
						&& (this->input_channel_ / this->group_ >= 16 && this->output_channel_ / this->group_ >= 16))
					{
						if (this->input_channel_ >= 16 && this->output_channel_ >= 16)
							conv3x3s1_winograd64_transform_kernel_neon5(this->weights_f32_[0], kernel_tm_, this->input_channel_, this->output_channel_, this->group_);
					}
					else if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 2 && this->stride_w_ == 2) && (this->dilation_h_ == 1 && this->dilation_w_ == 1))
					{
						conv3x3s2_transform_kernel_neon(this->weights_f32_[0], kernel_tm_, this->input_channel_, this->output_channel_);
						convolution_im2col_sgemm_transform_kernel_neon(this->weights_f32_[0], kernel_tm_gemm_, this->input_channel_, this->output_channel_, this->kernel_size_w_, this->kernel_size_h_);
					}
					else if ((this->kernel_size_h_ == 1 && this->kernel_size_w_ == 1) && (this->stride_h_ == 1 && this->stride_w_ == 1) && (this->dilation_h_ == 1 && this->dilation_w_ == 1))
					{
						if(this->input_channel_ >= 64 && this->output_channel_ >= 64)
							convolution_im2col_sgemm_transform_kernel_neon(this->weights_f32_[0], kernel_tm_gemm_, this->input_channel_, this->output_channel_, this->kernel_size_w_, this->kernel_size_h_);
					}
					else
					{
						convolution_im2col_sgemm_transform_kernel_neon(this->weights_f32_[0], kernel_tm_gemm_, this->input_channel_, this->output_channel_, this->kernel_size_w_, this->kernel_size_h_);
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}
			else
			{
				size_t align_data_size = (this->weight_data_size_ + 4 - 1) & -4;
				this->weights_i8_.push_back(std::shared_ptr<memory::tensor<signed char>>(new memory::tensor<signed char>(align_data_size, this->params_.device_, memory::NCHW, nullptr)));
				for (size_t i = 0; i < align_data_size; i++)
				{
					this->weights_i8_[0]->mutable_cpu_data()[i] = u(e);
				}
				mem += align_data_size;
				if (this->bias_term_)
				{
					this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(1, this->params_.device_, memory::NCHW, nullptr)));
					this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(this->output_channel_, this->params_.device_, memory::NCHW, nullptr)));
					for (size_t i = 0; i < this->output_channel_; i++)
					{
						this->weights_f32_[1]->mutable_cpu_data()[i] = n(e);
					}
					mem += this->output_channel_ * sizeof(float);
				}
				this->weights_scaletable_i8_.resize(this->output_channel_);
				for (size_t i = 0; i < this->output_channel_; i++)
				{
					this->weights_scaletable_i8_[i] = n(e);
				}
				this->featmap_scaletable_i8_.resize(1);
				this->featmap_scaletable_i8_[0] = n(e);
				mem += (this->output_channel_ + 1) * sizeof(float);

				if (this->group_ == 1)
				{
					if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 1 && this->stride_w_ == 1))
					{
						conv3x3s1_winograd43_transform_kernel_int8_neon();
					}
					else
					{
						conv_im2col_sgemm_transform_kernel_int8_neon();
					}
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}
			return mem;
		}

		template<typename Dtype>
		void operation_convolution_arm<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms, std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			CHECK_EQ(tops.size(), 1);
			CHECK_EQ(this->dilation_h_, 1);
			CHECK_EQ(this->dilation_w_, 1);

			int n = bottoms[0]->num();
			int w = bottoms[0]->width();
			int h = bottoms[0]->height();
			std::shared_ptr<memory::tensor<float> > bottom_bordered = bottoms[0];

			if (this->pad_left_ > 0 || this->pad_top_ > 0 || this->pad_right_ > 0 || this->pad_bottom_ > 0)
			{
				make_border(bottoms[0], bottom_bordered, this->pad_top_, this->pad_bottom_, this->pad_left_, this->pad_right_);
				w = bottom_bordered->width();
				h = bottom_bordered->height();
			}
			int outw = (w - this->kernel_size_w_) / this->stride_w_ + 1;
			int outh = (h - this->kernel_size_h_) / this->stride_h_ + 1;

			tops[0].reset(new memory::tensor<float>(std::vector<int> {n, this->output_channel_, outh, outw }, -1, memory::NCHW));

			memory::orderType order = bottoms[0]->order();
			if (!((order == memory::NCHW) || (order == memory::NHWC)))
			{
				NOT_IMPLEMENTED;
			}

			if (order == memory::NHWC)
			{
				bottom_bordered->convert_order();
			}

			if (this->group_ == 1)
			{
				if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 1 && this->stride_w_ == 1) && (this->dilation_h_ == 1 && this->dilation_w_ == 1))
				{
					if ((w <= 120 && h <= 120) && (this->input_channel_ >= 16 && this->output_channel_ >= 16))
						conv3x3s1_winograd64_neon5(bottom_bordered, tops[0], kernel_tm_, this->weights_f32_[1]);
					else
						conv3x3s1_neon(bottom_bordered, tops[0], this->weights_f32_[0], this->weights_f32_[1]);
				}
				else if ((this->kernel_size_h_ == 1 && this->kernel_size_w_ == 1) && (this->stride_h_ == 1 && this->stride_w_ == 1) && (this->dilation_h_ == 1 && this->dilation_w_ == 1))
				{
					if (this->input_channel_ >= 64 && this->output_channel_ >= 64)
						conv1x1s1_sgemm_neon(bottom_bordered, tops[0], kernel_tm_gemm_, this->weights_f32_[1]);
					else
						conv1x1s1_neon(bottom_bordered, tops[0], this->weights_f32_[0], this->weights_f32_[1]);
				}
				else if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 2 && this->stride_w_ == 2))
				{
					if (!(outw >= 8 && outh >= 8))
					{
						convolution_im2col_sgemm_neon(bottom_bordered, tops[0], kernel_tm_gemm_, this->weights_f32_[1], 
							this->kernel_size_w_, this->kernel_size_h_, this->dilation_w_, this->dilation_h_, this->stride_w_, this->stride_h_);
					}
					else
					{
						conv3x3s2_packed_neon(bottom_bordered, tops[0], kernel_tm_, this->weights_f32_[1]);
					}
				}
				else
				{
					convolution_im2col_sgemm_neon(bottom_bordered, tops[0], kernel_tm_gemm_, this->weights_f32_[1],
						this->kernel_size_w_, this->kernel_size_h_, this->dilation_w_, this->dilation_h_, this->stride_w_, this->stride_h_);
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			this->suffix_activation_cpu_f32(tops);
		}

		template<typename Dtype>
		void operation_convolution_arm<Dtype>::forward_cpu_i8(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			CHECK_EQ(tops.size(), 1);
			CHECK_EQ(this->dilation_h_, 1);
			CHECK_EQ(this->dilation_w_, 1);

			memory::orderType order = bottoms[0]->order();
			if (!((order == memory::NCHW) || (order == memory::NHWC)))
			{
				NOT_IMPLEMENTED;
			}

			int n = bottoms[0]->num();
			int w = bottoms[0]->width();
			int h = bottoms[0]->height();

			std::shared_ptr<memory::tensor<int8_t> > bottom_unbordered;

			this->quantize_float32_to_int8(bottoms[0], bottom_unbordered);

			if (order == memory::NHWC)
			{
				bottom_unbordered->convert_order();
			}

			std::shared_ptr<memory::tensor<int8_t> > bottom_bordered = bottom_unbordered;
			if (this->pad_left_ > 0 || this->pad_top_ > 0 || this->pad_right_ > 0 || this->pad_bottom_ > 0)
			{
				make_border(bottom_unbordered, bottom_bordered, this->pad_top_, this->pad_bottom_, this->pad_left_, this->pad_right_);
				w = bottom_bordered->width();
				h = bottom_bordered->height();
			}
			int outw = (w - this->kernel_size_w_) / this->stride_w_ + 1;
			int outh = (h - this->kernel_size_h_) / this->stride_h_ + 1;

			std::shared_ptr<memory::tensor<int>> top_int32_(new memory::tensor<int>(std::vector<int> {n, this->output_channel_, outh, outw }, -1, memory::NCHW));

			if (this->group_ == 1)
			{
				if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 1 && this->stride_w_ == 1))
				{
					conv3x3s1_winograd43_int8_neon(bottom_bordered, top_int32_);
				}
				else if ((this->kernel_size_h_ == 1 && this->kernel_size_w_ == 1) && (this->stride_h_ == 2 && this->stride_w_ == 2))
				{
					conv1x1s2_int8_neon(bottom_bordered, top_int32_);
				}
				else if ((this->kernel_size_h_ == 1 && this->kernel_size_w_ == 1) && (this->stride_h_ == 1 && this->stride_w_ == 1))
				{
					conv1x1s1_sgemm_int8_neon(bottom_bordered, top_int32_);
				}
				else
				{
					conv_im2col_sgemm_int8_neon(bottom_bordered, top_int32_);
				}

				// dequantize
				tops[0].reset(new memory::tensor<float>(top_int32_->data_shape(), this->params_.device_, top_int32_->order()));
				this->dequantize_int32_to_float32(top_int32_, tops[0]);

				this->suffix_activation_cpu_f32(tops);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}



		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv3x3s1_winograd43_transform_kernel_int8_neon()
		{
			int inch = this->input_channel_;
			int outch = this->output_channel_;
			const int8_t* kernel = this->weights_i8_[0]->cpu_data();

			std::shared_ptr<memory::tensor<short>> kernel_temp(new memory::tensor<short>(std::vector<int>{ 1, outch, inch, 6 * 6}, -1, memory::NCHW));
			short* kernel_temp_data = kernel_temp->mutable_cpu_data();

			// G
			// const float ktm[6][3] = {
			//     {  1.0f/4,     0.0f,    0.0f},
			//     { -1.0f/6,  -1.0f/6, -1.0f/6},
			//     { -1.0f/6,   1.0f/6, -1.0f/6},
			//     { 1.0f/24,  1.0f/12,  1.0f/6},
			//     { 1.0f/24, -1.0f/12,  1.0f/6},
			//     {    0.0f,     0.0f,    1.0f}
			// };
			const short ktm[6][3] = {
				{6, 0, 0},
				{-4, -4, -4},
				{-4, 4, -4},
				{1, 2, 4},
				{1, -2, 4},
				{0, 0, 6}
			};

			for (int p = 0; p < outch; p++)
			{
				for (int q = 0; q < inch; q++)
				{
					const signed char* kernel0 = (const signed char*)kernel + p * inch * 9 + q * 9;
					short* kernel_tm0 = kernel_temp_data + 6 * 6 * inch * p + 6 * 6 * q;

					// transform kernel
					const signed char* k0 = kernel0;
					const signed char* k1 = kernel0 + 3;
					const signed char* k2 = kernel0 + 6;

					// h
					short tmp[6][3];
					for (int i = 0; i < 6; i++)
					{
						tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
						tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
						tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
					}

					// U
					for (int j = 0; j < 6; j++)
					{
						short* tmpp = &tmp[j][0];

						for (int i = 0; i < 6; i++)
						{
							kernel_tm0[j * 6 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
						}
					}
				}
			}

			for (int r = 0; r < 9; r++)
			{
				std::shared_ptr<memory::tensor<short>> kernel_tm_test(new memory::tensor<short>(std::vector<int>{1, outch / 8 + (outch % 8) / 4 + outch % 4, inch, 4 * 8}, -1, memory::NCHW));

				int p = 0;
				for (; p + 7 < outch; p += 8)
				{
					const short* kernel0 = (const short*)kernel_temp_data + p * kernel_temp->count(2, 4);
					const short* kernel1 = (const short*)kernel_temp_data + (p + 1) * kernel_temp->count(2, 4);
					const short* kernel2 = (const short*)kernel_temp_data + (p + 2) * kernel_temp->count(2, 4);
					const short* kernel3 = (const short*)kernel_temp_data + (p + 3) * kernel_temp->count(2, 4);
					const short* kernel4 = (const short*)kernel_temp_data + (p + 4) * kernel_temp->count(2, 4);
					const short* kernel5 = (const short*)kernel_temp_data + (p + 5) * kernel_temp->count(2, 4);
					const short* kernel6 = (const short*)kernel_temp_data + (p + 6) * kernel_temp->count(2, 4);
					const short* kernel7 = (const short*)kernel_temp_data + (p + 7) * kernel_temp->count(2, 4);

					short* ktmp = kernel_tm_test->mutable_cpu_data() + p / 8 * kernel_tm_test->count(2,4);

					for (int q = 0; q < inch; q++)
					{
						ktmp[0] = kernel0[r * 4 + 0];
						ktmp[1] = kernel0[r * 4 + 1];
						ktmp[2] = kernel0[r * 4 + 2];
						ktmp[3] = kernel0[r * 4 + 3];

						ktmp[4] = kernel1[r * 4 + 0];
						ktmp[5] = kernel1[r * 4 + 1];
						ktmp[6] = kernel1[r * 4 + 2];
						ktmp[7] = kernel1[r * 4 + 3];

						ktmp[8] = kernel2[r * 4 + 0];
						ktmp[9] = kernel2[r * 4 + 1];
						ktmp[10] = kernel2[r * 4 + 2];
						ktmp[11] = kernel2[r * 4 + 3];

						ktmp[12] = kernel3[r * 4 + 0];
						ktmp[13] = kernel3[r * 4 + 1];
						ktmp[14] = kernel3[r * 4 + 2];
						ktmp[15] = kernel3[r * 4 + 3];

						ktmp[16] = kernel4[r * 4 + 0];
						ktmp[17] = kernel4[r * 4 + 1];
						ktmp[18] = kernel4[r * 4 + 2];
						ktmp[19] = kernel4[r * 4 + 3];

						ktmp[20] = kernel5[r * 4 + 0];
						ktmp[21] = kernel5[r * 4 + 1];
						ktmp[22] = kernel5[r * 4 + 2];
						ktmp[23] = kernel5[r * 4 + 3];

						ktmp[24] = kernel6[r * 4 + 0];
						ktmp[25] = kernel6[r * 4 + 1];
						ktmp[26] = kernel6[r * 4 + 2];
						ktmp[27] = kernel6[r * 4 + 3];

						ktmp[28] = kernel7[r * 4 + 0];
						ktmp[29] = kernel7[r * 4 + 1];
						ktmp[30] = kernel7[r * 4 + 2];
						ktmp[31] = kernel7[r * 4 + 3];

						ktmp += 32;
						kernel0 += 36;
						kernel1 += 36;
						kernel2 += 36;
						kernel3 += 36;
						kernel4 += 36;
						kernel5 += 36;
						kernel6 += 36;
						kernel7 += 36;
					}
				}

				for (; p + 3 < outch; p += 4)
				{
					const short* kernel0 = (const short*)kernel_temp_data + p * kernel_temp->count(2, 4);
					const short* kernel1 = (const short*)kernel_temp_data + (p + 1) * kernel_temp->count(2, 4);
					const short* kernel2 = (const short*)kernel_temp_data + (p + 2) * kernel_temp->count(2, 4);
					const short* kernel3 = (const short*)kernel_temp_data + (p + 3) * kernel_temp->count(2, 4);

					short* ktmp = kernel_tm_test->mutable_cpu_data() + p / 8 + (p % 8) / 4 * kernel_tm_test->count(2, 4);

					for (int q = 0; q < inch; q++)
					{
						ktmp[0] = kernel0[r * 4 + 0];
						ktmp[1] = kernel0[r * 4 + 1];
						ktmp[2] = kernel0[r * 4 + 2];
						ktmp[3] = kernel0[r * 4 + 3];

						ktmp[4] = kernel1[r * 4 + 0];
						ktmp[5] = kernel1[r * 4 + 1];
						ktmp[6] = kernel1[r * 4 + 2];
						ktmp[7] = kernel1[r * 4 + 3];

						ktmp[8] = kernel2[r * 4 + 0];
						ktmp[9] = kernel2[r * 4 + 1];
						ktmp[10] = kernel2[r * 4 + 2];
						ktmp[11] = kernel2[r * 4 + 3];

						ktmp[12] = kernel3[r * 4 + 0];
						ktmp[13] = kernel3[r * 4 + 1];
						ktmp[14] = kernel3[r * 4 + 2];
						ktmp[15] = kernel3[r * 4 + 3];

						ktmp += 16;
						kernel0 += 36;
						kernel1 += 36;
						kernel2 += 36;
						kernel3 += 36;
					}
				}

				for (; p < outch; p++)
				{
					const short* kernel0 = (const short*)kernel_temp_data + p * kernel_temp->count(2, 4);

					short* ktmp = kernel_tm_test->mutable_cpu_data() + (p / 8 + (p % 8) / 4 + p % 4) * kernel_tm_test->count(2, 4);

					for (int q = 0; q < inch; q++)
					{
						ktmp[0] = kernel0[r * 4 + 0];
						ktmp[1] = kernel0[r * 4 + 1];
						ktmp[2] = kernel0[r * 4 + 2];
						ktmp[3] = kernel0[r * 4 + 3];

						ktmp += 4;
						kernel0 += 36;
					}
				}
				kernel_tm_winograd_int8_.push_back(kernel_tm_test);
			}
		}

		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv_im2col_sgemm_transform_kernel_int8_neon()
		{
			int inch = this->input_channel_;
			int outch = this->output_channel_;
			const signed char* kernel = this->weights_i8_[0]->cpu_data();

			const int maxk = this->kernel_size_w_ * this->kernel_size_h_;

#if __ARM_NEON
			// interleave
			// src = maxk-inch-outch
			// dst = 8a-4b-maxk-inch/8a-outch/4b
			// dst = 4a-4b-2-maxk-inch/8a-outch/4b (arm82)

			if (outch >= 4)
			{
				if (inch >= 8)
					kernel_tm_gemm_int8_.reset(new memory::tensor<int8_t>(std::vector<int>{1, outch / 4 + outch % 4, inch / 8 + (inch % 8) / 4 + inch % 4, 32 * maxk}, this->params_.device_, memory::NCHW));
				else if (inch >= 4)
					kernel_tm_gemm_int8_.reset(new memory::tensor<int8_t>(std::vector<int>{1, outch / 4 + outch % 4, inch / 4 + inch % 4, 16 * maxk}, this->params_.device_, memory::NCHW));
				else
					kernel_tm_gemm_int8_.reset(new memory::tensor<int8_t>(std::vector<int>{1, outch / 4 + outch % 4, inch, 4 * maxk}, this->params_.device_, memory::NCHW));
			}
			else
			{
				if (inch >= 8)
					kernel_tm_gemm_int8_.reset(new memory::tensor<int8_t>(std::vector<int>{1, outch, inch / 8 + (inch % 8) / 4 + inch % 4, 8 * maxk}, this->params_.device_, memory::NCHW));
				else if (inch >= 4)
					kernel_tm_gemm_int8_.reset(new memory::tensor<int8_t>(std::vector<int>{1, outch, inch / 4 + inch % 4, 4 * maxk}, this->params_.device_, memory::NCHW));
				else
					kernel_tm_gemm_int8_.reset(new memory::tensor<int8_t>(std::vector<int>{1, outch, inch, 1 * maxk}, this->params_.device_, memory::NCHW));
			}

			int q = 0;
			for (; q + 3 < outch; q += 4)
			{
				signed char* g00 = kernel_tm_gemm_int8_->mutable_cpu_data() + q / 4 * kernel_tm_gemm_int8_->count(2, 4);

				int p = 0;
				for (; p + 7 < inch; p += 8)
				{
					for (int k = 0; k < maxk; k++)
					{
#if __ARM_FEATURE_DOTPROD
						for (int i = 0; i < 4; i++)
						{
							for (int j = 0; j < 4; j++)
							{
								const signed char* k00 = kernel + (q + i) * inch * maxk + (p + j) * maxk;

								g00[0] = k00[k];

								g00++;
							}
						}
						for (int i = 0; i < 4; i++)
						{
							for (int j = 4; j < 8; j++)
							{
								const signed char* k00 = kernel + (q + i) * inch * maxk + (p + j) * maxk;

								g00[0] = k00[k];

								g00++;
							}
						}
#else
						for (int i = 0; i < 4; i++)
						{
							for (int j = 0; j < 8; j++)
							{
								const signed char* k00 = kernel + (q + i) * inch * maxk + (p + j) * maxk;

								g00[0] = k00[k];

								g00++;
							}
						}
#endif
					}
				}
				for (; p + 3 < inch; p += 4)
				{
					for (int k = 0; k < maxk; k++)
					{
						for (int i = 0; i < 4; i++)
						{
							for (int j = 0; j < 4; j++)
							{
								const signed char* k00 = kernel + (q + i) * inch * maxk + (p + j) * maxk;

								g00[0] = k00[k];

								g00++;
							}
						}
					}
				}
				for (; p < inch; p++)
				{
					for (int k = 0; k < maxk; k++)
					{
						for (int i = 0; i < 4; i++)
						{
							const signed char* k00 = kernel + (q + i) * inch * maxk + (p)*maxk;

							g00[0] = k00[k];

							g00++;
						}
					}
				}
			}
			// TODO unroll 2
			for (; q < outch; q++)
			{
				signed char* g00 = kernel_tm_gemm_int8_->mutable_cpu_data() + (q / 4 + q % 4) * kernel_tm_gemm_int8_->count(2, 4);

				int p = 0;
				for (; p + 7 < inch; p += 8)
				{
					for (int k = 0; k < maxk; k++)
					{
						for (int j = 0; j < 8; j++)
						{
							const signed char* k00 = kernel + (q)*inch * maxk + (p + j) * maxk;

							g00[0] = k00[k];

							g00++;
						}
					}
				}
				for (; p + 3 < inch; p += 4)
				{
					for (int k = 0; k < maxk; k++)
					{
						for (int j = 0; j < 4; j++)
						{
							const signed char* k00 = kernel + (q)*inch * maxk + (p + j) * maxk;

							g00[0] = k00[k];

							g00++;
						}
					}
				}
				for (; p < inch; p++)
				{
					for (int k = 0; k < maxk; k++)
					{
						const signed char* k00 = kernel + q * inch * maxk + (p)*maxk;

						g00[0] = k00[k];

						g00++;
					}
				}
			}
#else  // __ARM_NEON
			kernel_tm_gemm_int8_ = this->weights_i8_[0];
			kernel_tm_gemm_int8_->reshape(std::vector<int>{ 1, outch, inch, maxk });
#endif // __ARM_NEON
		}

		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv3x3s1_winograd43_int8_neon(const std::shared_ptr<memory::tensor<int8_t>>& bottom, std::shared_ptr<memory::tensor<int>>& top)
		{
			int num = bottom->num();
			int w = bottom->width();
			int h = bottom->height();
			int inch = bottom->channels();

			int outw = top->width();
			int outh = top->height();
			int outch = top->channels();

			// pad to 4n+2, winograd F(4,3)
			std::shared_ptr<memory::tensor<int8_t>> bottom_bordered;

			outw = (outw + 3) / 4 * 4;
			outh = (outh + 3) / 4 * 4;

			w = outw + 2;
			h = outh + 2;

			make_border(bottom, bottom_bordered, 0, h - bottom->height(), 0, w - bottom->width());
			int bottom_bordered_cstep = bottom_bordered->count(2, 4);

			// BEGIN transform input
			std::shared_ptr<memory::tensor<short>> bottom_tm;
			{
				int w_tm = outw / 4 * 6;
				int h_tm = outh / 4 * 6;

				int nColBlocks = h_tm / 6; // may be the block num in Feathercnn
				int nRowBlocks = w_tm / 6;

				const int tiles = nColBlocks * nRowBlocks;

				bottom_tm = std::make_shared<memory::tensor<short>>(std::vector<int>{num, tiles * 9, inch, 4}, -1, memory::NCHW);
				int bottom_tm_cstep = bottom_tm->count(2, 4);
				for (size_t num_i = 0; num_i < num; num_i++)
				{
					const int8_t* bottom_bordered_data = bottom_bordered->cpu_data() + num_i * bottom_bordered->count(1, 4);
					short* bottom_tm_data = bottom_tm->mutable_cpu_data() + num_i * bottom_tm->count(1, 4);

					// BT
					// const float itm[4][4] = {
					//     {4.0f, 0.0f, -5.0f, 0.0f, 1.0f, 0.0f},
					//     {0.0f,-4.0f, -4.0f, 1.0f, 1.0f, 0.0f},
					//     {0.0f, 4.0f, -4.0f,-1.0f, 1.0f, 0.0f},
					//     {0.0f,-2.0f, -1.0f, 2.0f, 1.0f, 0.0f},
					//     {0.0f, 2.0f, -1.0f,-2.0f, 1.0f, 0.0f},
					//     {0.0f, 4.0f,  0.0f,-5.0f, 0.0f, 1.0f}
					// };

					// 0 =	4 * r00  - 5 * r02	+ r04
					// 1 = -4 * (r01 + r02)  + r03 + r04
					// 2 =	4 * (r01 - r02)  - r03 + r04
					// 3 = -2 * r01 - r02 + 2 * r03 + r04
					// 4 =	2 * r01 - r02 - 2 * r03 + r04
					// 5 =	4 * r01 - 5 * r03 + r05

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int q = 0; q < inch; q++)
					{
						const signed char* img = bottom_bordered_data + q * bottom_bordered_cstep;

						for (int j = 0; j < nColBlocks; j++)
						{
							const signed char* r0 = img + w * j * 4;
							const signed char* r1 = r0 + w;
							const signed char* r2 = r1 + w;
							const signed char* r3 = r2 + w;
							const signed char* r4 = r3 + w;
							const signed char* r5 = r4 + w;

							for (int i = 0; i < nRowBlocks; i++)
							{
								short* out_tm0 = bottom_tm_data + bottom_tm_cstep * (tiles * 0 + j * nRowBlocks + i) + q * 4;
								short* out_tm1 = bottom_tm_data + bottom_tm_cstep * (tiles * 1 + j * nRowBlocks + i) + q * 4;
								short* out_tm2 = bottom_tm_data + bottom_tm_cstep * (tiles * 2 + j * nRowBlocks + i) + q * 4;
								short* out_tm3 = bottom_tm_data + bottom_tm_cstep * (tiles * 3 + j * nRowBlocks + i) + q * 4;
								short* out_tm4 = bottom_tm_data + bottom_tm_cstep * (tiles * 4 + j * nRowBlocks + i) + q * 4;
								short* out_tm5 = bottom_tm_data + bottom_tm_cstep * (tiles * 5 + j * nRowBlocks + i) + q * 4;
								short* out_tm6 = bottom_tm_data + bottom_tm_cstep * (tiles * 6 + j * nRowBlocks + i) + q * 4;
								short* out_tm7 = bottom_tm_data + bottom_tm_cstep * (tiles * 7 + j * nRowBlocks + i) + q * 4;
								short* out_tm8 = bottom_tm_data + bottom_tm_cstep * (tiles * 8 + j * nRowBlocks + i) + q * 4;
#if __ARM_NEON
								int8x8_t _d0, _d1, _d2, _d3, _d4, _d5;
								int16x8_t _w0, _w1, _w2, _w3, _w4, _w5;
								int16x8_t _t0, _t1, _t2, _t3, _t4, _t5;
								int16x8_t _n0, _n1, _n2, _n3, _n4, _n5;
								// load
								_d0 = vld1_s8(r0);
								_d1 = vld1_s8(r1);
								_d2 = vld1_s8(r2);
								_d3 = vld1_s8(r3);
								_d4 = vld1_s8(r4);
								_d5 = vld1_s8(r5);

								int8x8_t _1_n = vdup_n_s8(-1);
								int8x8_t _2_p = vdup_n_s8(2);
								int8x8_t _2_n = vdup_n_s8(-2);
								int8x8_t _4_p = vdup_n_s8(4);
								int8x8_t _4_n = vdup_n_s8(-4);
								int8x8_t _5_n = vdup_n_s8(-5);

								int16x8_t _1_n_s16 = vdupq_n_s16(-1);
								int16x8_t _2_p_s16 = vdupq_n_s16(2);
								int16x8_t _2_n_s16 = vdupq_n_s16(-2);
								int16x8_t _4_p_s16 = vdupq_n_s16(4);
								int16x8_t _4_n_s16 = vdupq_n_s16(-4);
								int16x8_t _5_n_s16 = vdupq_n_s16(-5);
								// w = B_t * d
								_w0 = vmull_s8(_d0, _4_p);
								_w0 = vmlal_s8(_w0, _d2, _5_n);
								_w0 = vaddw_s8(_w0, _d4);

								_w1 = vmull_s8(_d1, _4_n);
								_w1 = vmlal_s8(_w1, _d2, _4_n);
								_w1 = vaddw_s8(_w1, _d3);
								_w1 = vaddw_s8(_w1, _d4);

								_w2 = vmull_s8(_d1, _4_p);
								_w2 = vmlal_s8(_w2, _d2, _4_n);
								_w2 = vmlal_s8(_w2, _d3, _1_n);
								_w2 = vaddw_s8(_w2, _d4);

								_w3 = vmull_s8(_d1, _2_n);
								_w3 = vmlal_s8(_w3, _d2, _1_n);
								_w3 = vmlal_s8(_w3, _d3, _2_p);
								_w3 = vaddw_s8(_w3, _d4);

								_w4 = vmull_s8(_d1, _2_p);
								_w4 = vmlal_s8(_w4, _d2, _1_n);
								_w4 = vmlal_s8(_w4, _d3, _2_n);
								_w4 = vaddw_s8(_w4, _d4);

								_w5 = vmull_s8(_d1, _4_p);
								_w5 = vmlal_s8(_w5, _d3, _5_n);
								_w5 = vaddw_s8(_w5, _d5);
								// transpose d to d_t
								{
									_t0[0] = _w0[0];
									_t1[0] = _w0[1];
									_t2[0] = _w0[2];
									_t3[0] = _w0[3];
									_t4[0] = _w0[4];
									_t5[0] = _w0[5];
									_t0[1] = _w1[0];
									_t1[1] = _w1[1];
									_t2[1] = _w1[2];
									_t3[1] = _w1[3];
									_t4[1] = _w1[4];
									_t5[1] = _w1[5];
									_t0[2] = _w2[0];
									_t1[2] = _w2[1];
									_t2[2] = _w2[2];
									_t3[2] = _w2[3];
									_t4[2] = _w2[4];
									_t5[2] = _w2[5];
									_t0[3] = _w3[0];
									_t1[3] = _w3[1];
									_t2[3] = _w3[2];
									_t3[3] = _w3[3];
									_t4[3] = _w3[4];
									_t5[3] = _w3[5];
									_t0[4] = _w4[0];
									_t1[4] = _w4[1];
									_t2[4] = _w4[2];
									_t3[4] = _w4[3];
									_t4[4] = _w4[4];
									_t5[4] = _w4[5];
									_t0[5] = _w5[0];
									_t1[5] = _w5[1];
									_t2[5] = _w5[2];
									_t3[5] = _w5[3];
									_t4[5] = _w5[4];
									_t5[5] = _w5[5];
								}
								// d = B_t * d_t
								_n0 = vmulq_s16(_t0, _4_p_s16);
								_n0 = vmlaq_s16(_n0, _t2, _5_n_s16);
								_n0 = vaddq_s16(_n0, _t4);

								_n1 = vmulq_s16(_t1, _4_n_s16);
								_n1 = vmlaq_s16(_n1, _t2, _4_n_s16);
								_n1 = vaddq_s16(_n1, _t3);
								_n1 = vaddq_s16(_n1, _t4);

								_n2 = vmulq_s16(_t1, _4_p_s16);
								_n2 = vmlaq_s16(_n2, _t2, _4_n_s16);
								_n2 = vmlaq_s16(_n2, _t3, _1_n_s16);
								_n2 = vaddq_s16(_n2, _t4);

								_n3 = vmulq_s16(_t1, _2_n_s16);
								_n3 = vmlaq_s16(_n3, _t2, _1_n_s16);
								_n3 = vmlaq_s16(_n3, _t3, _2_p_s16);
								_n3 = vaddq_s16(_n3, _t4);

								_n4 = vmulq_s16(_t1, _2_p_s16);
								_n4 = vmlaq_s16(_n4, _t2, _1_n_s16);
								_n4 = vmlaq_s16(_n4, _t3, _2_n_s16);
								_n4 = vaddq_s16(_n4, _t4);

								_n5 = vmulq_s16(_t1, _4_p_s16);
								_n5 = vmlaq_s16(_n5, _t3, _5_n_s16);
								_n5 = vaddq_s16(_n5, _t5);
								// save to out_tm
								out_tm0[0] = _n0[0];
								out_tm0[1] = _n0[1];
								out_tm0[2] = _n0[2];
								out_tm0[3] = _n0[3];
								out_tm1[0] = _n0[4];
								out_tm1[1] = _n0[5];
								out_tm1[2] = _n1[0];
								out_tm1[3] = _n1[1];
								out_tm2[0] = _n1[2];
								out_tm2[1] = _n1[3];
								out_tm2[2] = _n1[4];
								out_tm2[3] = _n1[5];

								out_tm3[0] = _n2[0];
								out_tm3[1] = _n2[1];
								out_tm3[2] = _n2[2];
								out_tm3[3] = _n2[3];
								out_tm4[0] = _n2[4];
								out_tm4[1] = _n2[5];
								out_tm4[2] = _n3[0];
								out_tm4[3] = _n3[1];
								out_tm5[0] = _n3[2];
								out_tm5[1] = _n3[3];
								out_tm5[2] = _n3[4];
								out_tm5[3] = _n3[5];

								out_tm6[0] = _n4[0];
								out_tm6[1] = _n4[1];
								out_tm6[2] = _n4[2];
								out_tm6[3] = _n4[3];
								out_tm7[0] = _n4[4];
								out_tm7[1] = _n4[5];
								out_tm7[2] = _n5[0];
								out_tm7[3] = _n5[1];
								out_tm8[0] = _n5[2];
								out_tm8[1] = _n5[3];
								out_tm8[2] = _n5[4];
								out_tm8[3] = _n5[5];
#else
								short d0[6], d1[6], d2[6], d3[6], d4[6], d5[6];
								short w0[6], w1[6], w2[6], w3[6], w4[6], w5[6];
								short t0[6], t1[6], t2[6], t3[6], t4[6], t5[6];

								// load
								for (int n = 0; n < 6; n++)
								{
									d0[n] = r0[n];
									d1[n] = r1[n];
									d2[n] = r2[n];
									d3[n] = r3[n];
									d4[n] = r4[n];
									d5[n] = r5[n];
								}
								// w = B_t * d
								for (int n = 0; n < 6; n++)
								{
									w0[n] = 4 * d0[n] - 5 * d2[n] + d4[n];
									w1[n] = -4 * d1[n] - 4 * d2[n] + d3[n] + d4[n];
									w2[n] = 4 * d1[n] - 4 * d2[n] - d3[n] + d4[n];
									w3[n] = -2 * d1[n] - d2[n] + 2 * d3[n] + d4[n];
									w4[n] = 2 * d1[n] - d2[n] - 2 * d3[n] + d4[n];
									w5[n] = 4 * d1[n] - 5 * d3[n] + d5[n];
								}
								// transpose d to d_t
								{
									t0[0] = w0[0];
									t1[0] = w0[1];
									t2[0] = w0[2];
									t3[0] = w0[3];
									t4[0] = w0[4];
									t5[0] = w0[5];
									t0[1] = w1[0];
									t1[1] = w1[1];
									t2[1] = w1[2];
									t3[1] = w1[3];
									t4[1] = w1[4];
									t5[1] = w1[5];
									t0[2] = w2[0];
									t1[2] = w2[1];
									t2[2] = w2[2];
									t3[2] = w2[3];
									t4[2] = w2[4];
									t5[2] = w2[5];
									t0[3] = w3[0];
									t1[3] = w3[1];
									t2[3] = w3[2];
									t3[3] = w3[3];
									t4[3] = w3[4];
									t5[3] = w3[5];
									t0[4] = w4[0];
									t1[4] = w4[1];
									t2[4] = w4[2];
									t3[4] = w4[3];
									t4[4] = w4[4];
									t5[4] = w4[5];
									t0[5] = w5[0];
									t1[5] = w5[1];
									t2[5] = w5[2];
									t3[5] = w5[3];
									t4[5] = w5[4];
									t5[5] = w5[5];
								}
								// d = B_t * d_t
								for (int n = 0; n < 6; n++)
								{
									d0[n] = 4 * t0[n] - 5 * t2[n] + t4[n];
									d1[n] = -4 * t1[n] - 4 * t2[n] + t3[n] + t4[n];
									d2[n] = 4 * t1[n] - 4 * t2[n] - t3[n] + t4[n];
									d3[n] = -2 * t1[n] - t2[n] + 2 * t3[n] + t4[n];
									d4[n] = 2 * t1[n] - t2[n] - 2 * t3[n] + t4[n];
									d5[n] = 4 * t1[n] - 5 * t3[n] + t5[n];
								}
								// save to out_tm
								{
									out_tm0[0] = d0[0];
									out_tm0[1] = d0[1];
									out_tm0[2] = d0[2];
									out_tm0[3] = d0[3];
									out_tm1[0] = d0[4];
									out_tm1[1] = d0[5];
									out_tm1[2] = d1[0];
									out_tm1[3] = d1[1];
									out_tm2[0] = d1[2];
									out_tm2[1] = d1[3];
									out_tm2[2] = d1[4];
									out_tm2[3] = d1[5];

									out_tm3[0] = d2[0];
									out_tm3[1] = d2[1];
									out_tm3[2] = d2[2];
									out_tm3[3] = d2[3];
									out_tm4[0] = d2[4];
									out_tm4[1] = d2[5];
									out_tm4[2] = d3[0];
									out_tm4[3] = d3[1];
									out_tm5[0] = d3[2];
									out_tm5[1] = d3[3];
									out_tm5[2] = d3[4];
									out_tm5[3] = d3[5];

									out_tm6[0] = d4[0];
									out_tm6[1] = d4[1];
									out_tm6[2] = d4[2];
									out_tm6[3] = d4[3];
									out_tm7[0] = d4[4];
									out_tm7[1] = d4[5];
									out_tm7[2] = d5[0];
									out_tm7[3] = d5[1];
									out_tm8[0] = d5[2];
									out_tm8[1] = d5[3];
									out_tm8[2] = d5[4];
									out_tm8[3] = d5[5];
								}
#endif // __ARM_NEON
								r0 += 4;
								r1 += 4;
								r2 += 4;
								r3 += 4;
								r4 += 4;
								r5 += 4;
							}
						}
					}
				}
			}

			// BEGIN dot
			std::shared_ptr<memory::tensor<int>> top_tm;
			{
				int w_tm = outw / 4 * 6;
				int h_tm = outh / 4 * 6;

				int nColBlocks = h_tm / 6; // may be the block num in Feathercnn
				int nRowBlocks = w_tm / 6;

				const int tiles = nColBlocks * nRowBlocks;

				top_tm = std::make_shared<memory::tensor<int>>(std::vector<int>{num, outch, tiles, 36}, -1, memory::NCHW);
				int top_tm_cstep = top_tm->count(2, 4);
				int bottom_tm_cstep = bottom_tm->count(2, 4);
				for (size_t num_i = 0; num_i < num; num_i++)
				{
					const short* bottom_tm_data = bottom_tm->cpu_data() + num_i * bottom_tm->count(1, 4);
					int* top_tm_data = top_tm->mutable_cpu_data() + num_i * top_tm->count(1, 4);

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int r = 0; r < 9; r++)
					{
						const short* kernel_tm_f43_int8_data = kernel_tm_winograd_int8_[r]->cpu_data();
						int kernel_tm_f43_int8_cstep = kernel_tm_winograd_int8_[r]->count(2, 4);

						int nn_outch = 0;
						int remain_outch_start = 0;

						nn_outch = outch >> 3;
						remain_outch_start = nn_outch << 3;

						for (int pp = 0; pp < nn_outch; pp++)
						{
							int p = pp * 8;

							int* output0_tm = top_tm_data + top_tm_cstep * (p);
							int* output1_tm = top_tm_data + top_tm_cstep * (p + 1);
							int* output2_tm = top_tm_data + top_tm_cstep * (p + 2);
							int* output3_tm = top_tm_data + top_tm_cstep * (p + 3);
							int* output4_tm = top_tm_data + top_tm_cstep * (p + 4);
							int* output5_tm = top_tm_data + top_tm_cstep * (p + 5);
							int* output6_tm = top_tm_data + top_tm_cstep * (p + 6);
							int* output7_tm = top_tm_data + top_tm_cstep * (p + 7);

							output0_tm = output0_tm + r * 4;
							output1_tm = output1_tm + r * 4;
							output2_tm = output2_tm + r * 4;
							output3_tm = output3_tm + r * 4;
							output4_tm = output4_tm + r * 4;
							output5_tm = output5_tm + r * 4;
							output6_tm = output6_tm + r * 4;
							output7_tm = output7_tm + r * 4;

							for (int i = 0; i < tiles; i++)
							{
								const short* kptr = kernel_tm_f43_int8_data + p / 8 * kernel_tm_f43_int8_cstep;
								const short* r0 = bottom_tm_data + bottom_tm_cstep * (tiles * r + i);
#if __ARM_NEON
#if __aarch64__
								asm volatile(
									// inch loop
									"eor    v0.16b, v0.16b, v0.16b    \n"
									"eor    v1.16b, v1.16b, v1.16b    \n"
									"eor    v2.16b, v2.16b, v2.16b    \n"
									"eor    v3.16b, v3.16b, v3.16b    \n"
									"eor    v4.16b, v4.16b, v4.16b    \n"
									"eor    v5.16b, v5.16b, v5.16b    \n"
									"eor    v6.16b, v6.16b, v6.16b    \n"
									"eor    v7.16b, v7.16b, v7.16b    \n"
									"mov    w4, %w20                  \n"

									"0:                               \n" // for (int q=0; q<inch; q++)
									"prfm    pldl1keep, [%9, #128]    \n" // _r0 = vld1_s16(r0);
									"ld1     {v8.4h}, [%8]            \n"
									"ld1     {v9.4h, v10.4h}, [%9]    \n" // _k01 = vld1q_s16(kptr);
									"add     %9, %9, #16              \n"
									"ld1     {v11.4h, v12.4h}, [%9]   \n" // _k23 = vld1q_s16(kptr+8);
									"add     %9, %9, #16              \n"
									"ld1     {v13.4h, v14.4h}, [%9]   \n" // _k45 = vld1q_s16(kptr+16);
									"add     %9, %9, #16              \n"
									"ld1     {v15.4h, v16.4h}, [%9]   \n" // _k67 = vld1q_s16(kptr+24);
									"add     %8, %8, #8               \n"
									"add     %9, %9, #16              \n"

									"subs    w4, w4, #1               \n"

									"smlal   v0.4s, v8.4h, v9.4h      \n" // sum0 += (a00-a03) * (k00-k03)
									"smlal   v1.4s, v8.4h, v10.4h     \n" // sum1 += (a00-a03) * (k10-k13)
									"smlal   v2.4s, v8.4h, v11.4h     \n" // sum2 += (a00-a03) * (k20-k23)
									"smlal   v3.4s, v8.4h, v12.4h     \n" // sum3 += (a00-a03) * (k30-k33)
									"smlal   v4.4s, v8.4h, v13.4h     \n" // sum4 += (a00-a03) * (k40-k43)
									"smlal   v5.4s, v8.4h, v14.4h     \n" // sum5 += (a00-a03) * (k50-k53)
									"smlal   v6.4s, v8.4h, v15.4h     \n" // sum6 += (a00-a03) * (k60-k63)
									"smlal   v7.4s, v8.4h, v16.4h     \n" // sum7 += (a00-a03) * (k70-k73)

									"bne     0b                       \n" // end for

									"st1     {v0.4s}, [%0]            \n" // store the result to memory
									"st1     {v1.4s}, [%1]            \n" //
									"st1     {v2.4s}, [%2]            \n" //
									"st1     {v3.4s}, [%3]            \n" //
									"st1     {v4.4s}, [%4]            \n" //
									"st1     {v5.4s}, [%5]            \n" //
									"st1     {v6.4s}, [%6]            \n" //
									"st1     {v7.4s}, [%7]            \n" //

									: "=r"(output0_tm), // %0
									"=r"(output1_tm), // %1
									"=r"(output2_tm), // %2
									"=r"(output3_tm), // %3
									"=r"(output4_tm), // %4
									"=r"(output5_tm), // %5
									"=r"(output6_tm), // %6
									"=r"(output7_tm), // %7
									"=r"(r0),         // %8
									"=r"(kptr)        // %9
									: "0"(output0_tm),
									"1"(output1_tm),
									"2"(output2_tm),
									"3"(output3_tm),
									"4"(output4_tm),
									"5"(output5_tm),
									"6"(output6_tm),
									"7"(output7_tm),
									"8"(r0),
									"9"(kptr),
									"r"(inch) // %20
									: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16");
#else
								asm volatile(
									// inch loop
									"vmov.s32    q0, #0           \n"
									"vmov.s32    q1, #0           \n"
									"vmov.s32    q2, #0           \n"
									"vmov.s32    q3, #0           \n"
									"vmov.s32    q4, #0           \n"
									"vmov.s32    q5, #0           \n"
									"vmov.s32    q6, #0           \n"
									"vmov.s32    q7, #0           \n"
									"mov         r4, %20          \n"

									"0:                           \n" // for (int q=0; q<inch; q++)
									"vld1.s16    {d16}, [%8]!     \n" // _r0 = vld1_s16(r0);  // input inch0
									"vld1.s16    {d18-d19}, [%9]  \n" // _k01 = vld1q_s16(kptr);
									"add         %9, #16          \n"
									"vld1.s16    {d20-d21}, [%9]  \n" // _k23 = vld1q_s16(kptr+8);
									"add         %9, #16          \n"
									"vld1.s16    {d22-d23}, [%9]  \n" // _k45 = vld1q_s16(kptr+16);
									"add         %9, #16          \n"
									"vld1.s16    {d24-d25}, [%9]  \n" // _k67 = vld1q_s16(kptr+24);
									"add         %9, #16          \n"

									"vmlal.s16   q0, d16, d18     \n" // sum0 += (a00-a03) * (k00-k03)
									"vmlal.s16   q1, d16, d19     \n" // sum1 += (a00-a03) * (k10-k13)
									"vmlal.s16   q2, d16, d20     \n" // sum2 += (a00-a03) * (k20-k23)
									"vmlal.s16   q3, d16, d21     \n" // sum3 += (a00-a03) * (k30-k33)
									"vmlal.s16   q4, d16, d22     \n" // sum4 += (a00-a03) * (k40-k43)
									"vmlal.s16   q5, d16, d23     \n" // sum5 += (a00-a03) * (k50-k53)
									"vmlal.s16   q6, d16, d24     \n" // sum6 += (a00-a03) * (k60-k63)
									"vmlal.s16   q7, d16, d25     \n" // sum7 += (a00-a03) * (k70-k73)

									"subs        r4, r4, #1       \n"
									"bne         0b               \n" // end for

									"vst1.s32    {d0-d1}, [%0]    \n" // store the result to memory
									"vst1.s32    {d2-d3}, [%1]    \n"
									"vst1.s32    {d4-d5}, [%2]    \n"
									"vst1.s32    {d6-d7}, [%3]    \n"
									"vst1.s32    {d8-d9}, [%4]    \n"
									"vst1.s32    {d10-d11}, [%5]  \n"
									"vst1.s32    {d12-d13}, [%6]  \n"
									"vst1.s32    {d14-d15}, [%7]  \n"

									: "=r"(output0_tm), // %0
									"=r"(output1_tm), // %1
									"=r"(output2_tm), // %2
									"=r"(output3_tm), // %3
									"=r"(output4_tm), // %4
									"=r"(output5_tm), // %5
									"=r"(output6_tm), // %6
									"=r"(output7_tm), // %7
									"=r"(r0),         // %8
									"=r"(kptr)        // %9
									: "0"(output0_tm),
									"1"(output1_tm),
									"2"(output2_tm),
									"3"(output3_tm),
									"4"(output4_tm),
									"5"(output5_tm),
									"6"(output6_tm),
									"7"(output7_tm),
									"8"(r0),
									"9"(kptr),
									"r"(inch) // %20
									: "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12");
#endif // __aarch64__
#else
								int sum0[4] = { 0 };
								int sum1[4] = { 0 };
								int sum2[4] = { 0 };
								int sum3[4] = { 0 };
								int sum4[4] = { 0 };
								int sum5[4] = { 0 };
								int sum6[4] = { 0 };
								int sum7[4] = { 0 };

								for (int q = 0; q < inch; q++)
								{
									for (int n = 0; n < 4; n++)
									{
										sum0[n] += (int)r0[n] * kptr[n];
										sum1[n] += (int)r0[n] * kptr[n + 4];
										sum2[n] += (int)r0[n] * kptr[n + 8];
										sum3[n] += (int)r0[n] * kptr[n + 12];
										sum4[n] += (int)r0[n] * kptr[n + 16];
										sum5[n] += (int)r0[n] * kptr[n + 20];
										sum6[n] += (int)r0[n] * kptr[n + 24];
										sum7[n] += (int)r0[n] * kptr[n + 28];
									}
									kptr += 32;
									r0 += 4;
								}

								for (int n = 0; n < 4; n++)
								{
									output0_tm[n] = sum0[n];
									output1_tm[n] = sum1[n];
									output2_tm[n] = sum2[n];
									output3_tm[n] = sum3[n];
									output4_tm[n] = sum4[n];
									output5_tm[n] = sum5[n];
									output6_tm[n] = sum6[n];
									output7_tm[n] = sum7[n];
								}
#endif // __ARM_NEON
								output0_tm += 36;
								output1_tm += 36;
								output2_tm += 36;
								output3_tm += 36;
								output4_tm += 36;
								output5_tm += 36;
								output6_tm += 36;
								output7_tm += 36;
							}
						}

						nn_outch = (outch - remain_outch_start) >> 2;

						for (int pp = 0; pp < nn_outch; pp++)
						{
							int p = remain_outch_start + pp * 4;

							int* output0_tm = top_tm_data + p * top_tm_cstep;
							int* output1_tm = top_tm_data + (p + 1) * top_tm_cstep;
							int* output2_tm = top_tm_data + (p + 2) * top_tm_cstep;
							int* output3_tm = top_tm_data + (p + 3) * top_tm_cstep;

							output0_tm = output0_tm + r * 4;
							output1_tm = output1_tm + r * 4;
							output2_tm = output2_tm + r * 4;
							output3_tm = output3_tm + r * 4;

							for (int i = 0; i < tiles; i++)
							{
								const short* kptr = kernel_tm_f43_int8_data + (p / 8 + (p % 8) / 4) * kernel_tm_f43_int8_cstep;
								const short* r0 = bottom_tm_data + (tiles * r + i) * bottom_tm_cstep;
#if __ARM_NEON
#if __aarch64__
								asm volatile(
									// inch loop
									"eor    v0.16b, v0.16b, v0.16b    \n"
									"eor    v1.16b, v1.16b, v1.16b    \n"
									"eor    v2.16b, v2.16b, v2.16b    \n"
									"eor    v3.16b, v3.16b, v3.16b    \n"
									"mov    w4, %w12                  \n"

									"0:                               \n" // for (int q=0; q<inch; q++)
									"prfm    pldl1keep, [%5, #128]    \n" // _r0 = vld1_s16(r0);  // input inch0
									"ld1     {v8.4h}, [%4]            \n"
									"ld1     {v9.4h, v10.4h}, [%5]    \n" // _k01 = vld1q_s16(kptr);
									"add     %5, %5, #16              \n"
									"ld1     {v11.4h, v12.4h}, [%5]   \n" // _k23 = vld1q_s16(kptr+8);
									"add     %4, %4, #8               \n"
									"add     %5, %5, #16              \n"

									"subs    w4, w4, #1               \n"

									"smlal   v0.4s, v8.4h, v9.4h      \n" // sum0 += (a00-a03) * (k00-k03)
									"smlal   v1.4s, v8.4h, v10.4h     \n" // sum1 += (a00-a03) * (k10-k13)
									"smlal   v2.4s, v8.4h, v11.4h     \n" // sum2 += (a00-a03) * (k20-k23)
									"smlal   v3.4s, v8.4h, v12.4h     \n" // sum3 += (a00-a03) * (k30-k33)

									"bne     0b                       \n" // end for

									"st1     {v0.4s}, [%0]            \n" // store the result to memory
									"st1     {v1.4s}, [%1]            \n" //
									"st1     {v2.4s}, [%2]            \n" //
									"st1     {v3.4s}, [%3]            \n" //

									: "=r"(output0_tm), // %0
									"=r"(output1_tm), // %1
									"=r"(output2_tm), // %2
									"=r"(output3_tm), // %3
									"=r"(r0),         // %4
									"=r"(kptr)        // %5
									: "0"(output0_tm),
									"1"(output1_tm),
									"2"(output2_tm),
									"3"(output3_tm),
									"4"(r0),
									"5"(kptr),
									"r"(inch) // %12
									: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12");
#else
								asm volatile(
									// inch loop
									"vmov.s32    q0, #0           \n"
									"vmov.s32    q1, #0           \n"
									"vmov.s32    q2, #0           \n"
									"vmov.s32    q3, #0           \n"
									"mov         r4, %12          \n"

									"0:                           \n" // for (int q=0; q<inch; q++)
									"vld1.s16    {d16}, [%4]!     \n" // _r0 = vld1_s16(r0);  // input inch0
									"vld1.s16    {d18-d19}, [%5]  \n" // _k01 = vld1q_s16(kptr);
									"add         %5, #16          \n"
									"vld1.s16    {d20-d21}, [%5]  \n" // _k23 = vld1q_s16(kptr+8);
									"add         %5, #16          \n"

									"vmlal.s16   q0, d16, d18     \n" // sum0 += (a00-a03) * (k00-k03)
									"vmlal.s16   q1, d16, d19     \n" // sum1 += (a00-a03) * (k10-k13)
									"vmlal.s16   q2, d16, d20     \n" // sum2 += (a00-a03) * (k20-k23)
									"vmlal.s16   q3, d16, d21     \n" // sum3 += (a00-a03) * (k30-k33)

									"subs        r4, r4, #1       \n"
									"bne         0b               \n" // end for

									"vst1.s32    {d0-d1}, [%0]    \n" // store the result to memory
									"vst1.s32    {d2-d3}, [%1]    \n"
									"vst1.s32    {d4-d5}, [%2]    \n"
									"vst1.s32    {d6-d7}, [%3]    \n"

									: "=r"(output0_tm), // %0
									"=r"(output1_tm), // %1
									"=r"(output2_tm), // %2
									"=r"(output3_tm), // %3
									"=r"(r0),         // %4
									"=r"(kptr)        // %5
									: "0"(output0_tm),
									"1"(output1_tm),
									"2"(output2_tm),
									"3"(output3_tm),
									"4"(r0),
									"5"(kptr),
									"r"(inch) // %12
									: "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10");
#endif // __aarch64__
#else
								int sum0[4] = { 0 };
								int sum1[4] = { 0 };
								int sum2[4] = { 0 };
								int sum3[4] = { 0 };

								for (int q = 0; q < inch; q++)
								{
									for (int n = 0; n < 4; n++)
									{
										sum0[n] += (int)r0[n] * kptr[n];
										sum1[n] += (int)r0[n] * kptr[n + 4];
										sum2[n] += (int)r0[n] * kptr[n + 8];
										sum3[n] += (int)r0[n] * kptr[n + 12];
									}
									kptr += 16;
									r0 += 4;
								}

								for (int n = 0; n < 4; n++)
								{
									output0_tm[n] = sum0[n];
									output1_tm[n] = sum1[n];
									output2_tm[n] = sum2[n];
									output3_tm[n] = sum3[n];
								}
#endif // __ARM_NEON
								output0_tm += 36;
								output1_tm += 36;
								output2_tm += 36;
								output3_tm += 36;
							}
						}

						remain_outch_start += nn_outch << 2;

						for (int p = remain_outch_start; p < outch; p++)
						{
							int* output0_tm = top_tm_data + p * top_tm_cstep;

							output0_tm = output0_tm + r * 4;

							for (int i = 0; i < tiles; i++)
							{
								const short* kptr = kernel_tm_f43_int8_data + (p / 8 + (p % 8) / 4 + p % 4) * kernel_tm_f43_int8_cstep;
								const short* r0 = bottom_tm_data + (tiles * r + i) * bottom_tm_cstep;
#if __ARM_NEON
#if __aarch64__
								asm volatile(
									// inch loop
									"eor    v0.16b, v0.16b, v0.16b    \n"
									"mov    w4, %w6                   \n"

									"0:                               \n" // for (int q=0; q<inch; q++)
									"ld1     {v8.4h}, [%1]            \n" // _r0 = vld1_s16(r0);  // input inch0
									"ld1     {v9.4h}, [%2]            \n" // _k0 = vld1q_s16(kptr);
									"add     %1, %1, #8               \n"
									"add     %2, %2, #8               \n"

									"subs    w4, w4, #1               \n"

									"smlal   v0.4s, v8.4h, v9.4h      \n" // sum0 += (a00-a03) * (k00-k03)

									"bne     0b                       \n" // end for

									"st1     {v0.4s}, [%0]            \n" // store the result to memory

									: "=r"(output0_tm), // %0
									"=r"(r0),         // %1
									"=r"(kptr)        // %2
									: "0"(output0_tm),
									"1"(r0),
									"2"(kptr),
									"r"(inch) // %6
									: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9");
#else
								asm volatile(
									// inch loop
									"vmov.s32    q0, #0           \n"
									"mov         r4, %6           \n"

									"0:                           \n" // for (int q=0; q<inch; q++)
									"vld1.s16    {d16}, [%1]      \n" // _r0 = vld1_s16(r0);  // input inch0
									"add         %1, #8           \n"
									"vld1.s16    {d18}, [%2]      \n" // _k0 = vld1q_s16(kptr);
									"add         %2, #8           \n"
									"vmlal.s16   q0, d16, d18     \n" // sum0 += (a00-a03) * (k00-k03)

									"subs        r4, r4, #1       \n"
									"bne         0b               \n" // end for

									"vst1.s32    {d0-d1}, [%0]    \n" // store the result to memory

									: "=r"(output0_tm), // %0
									"=r"(r0),         // %1
									"=r"(kptr)        // %2
									: "0"(output0_tm),
									"1"(r0),
									"2"(kptr),
									"r"(inch) // %6
									: "cc", "memory", "r4", "q0", "q8", "q9");
#endif // __aarch64__
#else  // __ARM_NEON
								int sum0[4] = { 0 };

								for (int q = 0; q < inch; q++)
								{
									for (int n = 0; n < 4; n++)
									{
										sum0[n] += (int)r0[n] * kptr[n];
									}
									kptr += 4;
									r0 += 4;
								}

								for (int n = 0; n < 4; n++)
								{
									output0_tm[n] = sum0[n];
								}
#endif // __ARM_NEON
								output0_tm += 36;
							}
						}

						// for (int p=0; p<outch; p++)
						// {
						//     Mat out0_tm = top_blob_tm.channel(p);
						//     const Mat kernel0_tm = kernel_tm.channel(p);

						//     for (int i=0; i<tiles; i++)
						//     {
						//         int* output0_tm = out0_tm.row<int>(i);

						//         int sum0[36] = {0};

						//         for (int q=0; q<inch; q++)
						//         {
						//             const short* r0 = bottom_blob_tm.channel(q).row<short>(i);
						//             const short* k0 = kernel0_tm.row<short>(q);

						//             for (int n=0; n<36; n++)
						//             {
						//                 sum0[n] += (int)r0[n] * k0[n];
						//             }
						//         }

						//         for (int n=0; n<36; n++)
						//         {
						//             output0_tm[n] = sum0[n];
						//         }
						//     }
						// }
					}
				}
			}
			// END dot

			// BEGIN transform output
			std::shared_ptr<memory::tensor<int>> top_bordered(new memory::tensor<int>(std::vector<int>{ num, outch, outh, outw }, this->params_.device_, memory::NCHW));
			{
				// AT
				// const float itm[4][6] = {
				//     {1.0f, 1.0f,  1.0f, 1.0f,  1.0f, 0.0f},
				//     {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.0f},
				//     {0.0f, 1.0f,  1.0f, 4.0f,  4.0f, 0.0f},
				//     {0.0f, 1.0f, -1.0f, 8.0f, -8.0f, 1.0f}
				// };

				// 0 =	r00 + r01 + r02 + r03 +	r04
				// 1 =		  r01 - r02 + 2 * (r03 - r04)
				// 2 =		  r01 + r02 + 4 * (r03 + r04)
				// 3 =		  r01 - r02 + 8 * (r03 - r04)  + r05

				int w_tm = outw / 4 * 6;
				int h_tm = outh / 4 * 6;

				int nColBlocks = h_tm / 6; // may be the block num in Feathercnn
				int nRowBlocks = w_tm / 6;

				int top_tm_cstep = top_tm->count(2, 4);
				int top_bordered_cstep = top_bordered->count(2, 4);
				for (size_t num_i = 0; num_i < num; num_i++)
				{
					const int* top_tm_data = top_tm->cpu_data() + num_i * top_tm->count(1, 4);
					int* top_bordered_data = top_bordered->mutable_cpu_data() + num_i * top_bordered->count(1, 4);
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int p = 0; p < outch; p++)
					{
						const int* out_tile = top_tm_data + p * top_tm_cstep;
						int* outRow0 = top_bordered_data + p * top_bordered_cstep;
						int* outRow1 = outRow0 + outw;
						int* outRow2 = outRow0 + outw * 2;
						int* outRow3 = outRow0 + outw * 3;

						for (int j = 0; j < nColBlocks; j++)
						{
							for (int i = 0; i < nRowBlocks; i++)
							{
#if __ARM_NEON
								int32x4_t _s0, _s1, _s2, _s3, _s4, _s5;
								int32x2_t _s0n, _s1n, _s2n, _s3n, _s4n, _s5n;
								int32x4_t _w0, _w3;
								int32x2_t _w0n, _w3n;
								int32x4_t _d0, _d1, _d2, _d3, _d4, _d5;
								int32x4_t _o0, _o1, _o2, _o3;
								// load
								_s0 = vld1q_s32(out_tile);
								_s0n = vld1_s32(out_tile + 4);
								_s1 = vld1q_s32(out_tile + 6);
								_s1n = vld1_s32(out_tile + 10);
								_s2 = vld1q_s32(out_tile + 12);
								_s2n = vld1_s32(out_tile + 16);
								_s3 = vld1q_s32(out_tile + 18);
								_s3n = vld1_s32(out_tile + 22);
								_s4 = vld1q_s32(out_tile + 24);
								_s4n = vld1_s32(out_tile + 28);
								_s5 = vld1q_s32(out_tile + 30);
								_s5n = vld1_s32(out_tile + 34);
								// w = A_T * W
								int32x2_t _tp0 = { 1, 4 };
								int32x2_t _tp1 = { 2, 8 };

								// 4*s5[n]
								int32x4_t _s5x4 = vshlq_n_s32(_s5, 2);
								int32x2_t _s5x4n = vshl_n_s32(_s5n, 2);

								int32x4_t _t1p2 = vaddq_s32(_s1, _s2);
								int32x2_t _t1p2n = vadd_s32(_s1n, _s2n);
								int32x4_t _t3p4 = vaddq_s32(_s3, _s4);
								int32x2_t _t3p4n = vadd_s32(_s3n, _s4n);
								int32x4_t _t1s2 = vsubq_s32(_s1, _s2);
								int32x2_t _t1s2n = vsub_s32(_s1n, _s2n);
								int32x4_t _t3s4 = vsubq_s32(_s3, _s4);
								int32x2_t _t3s4n = vsub_s32(_s3n, _s4n);

								_w0 = vaddq_s32(_s0, _t1p2);
								_w0n = vadd_s32(_s0n, _t1p2n);
								_w0 = vaddq_s32(_w0, _t3p4);
								_w0n = vadd_s32(_w0n, _t3p4n);
								_w0n = vmul_s32(_w0n, _tp0);

								// _w2,_w2n
								_t1p2 = vmlaq_lane_s32(_t1p2, _t3p4, _tp0, 1);
								_t1p2n = vmla_lane_s32(_t1p2n, _t3p4n, _tp0, 1);
								_t1p2n = vmul_s32(_t1p2n, _tp0);

								_w3 = vaddq_s32(_s5x4, _t1s2);
								_w3n = vadd_s32(_s5x4n, _t1s2n);
								_w3 = vmlaq_lane_s32(_w3, _t3s4, _tp1, 1);
								_w3n = vmla_lane_s32(_w3n, _t3s4n, _tp1, 1);
								_w3n = vmul_s32(_w3n, _tp0);

								// _w1, _w1n
								_t1s2 = vmlaq_lane_s32(_t1s2, _t3s4, _tp1, 0);
								_t1s2n = vmla_lane_s32(_t1s2n, _t3s4n, _tp1, 0);
								_t1s2n = vmul_s32(_t1s2n, _tp0);

								int32x4_t _w02n = vcombine_s32(_w0n, _t1p2n);
								int32x4_t _w13n = vcombine_s32(_t1s2n, _w3n);

								// transpose w to w_t
#if __aarch64__
								int32x4_t _wt0 = vtrn1q_s32(_w0, _t1s2);
								int32x4_t _wt1 = vtrn2q_s32(_w0, _t1s2);
								int32x4_t _wt2 = vtrn1q_s32(_t1p2, _w3);
								int32x4_t _wt3 = vtrn2q_s32(_t1p2, _w3);
								int64x2_t _dt0 = vtrn1q_s64(vreinterpretq_s64_s32(_wt0), vreinterpretq_s64_s32(_wt2));
								int64x2_t _dt2 = vtrn2q_s64(vreinterpretq_s64_s32(_wt0), vreinterpretq_s64_s32(_wt2));
								int64x2_t _dt1 = vtrn1q_s64(vreinterpretq_s64_s32(_wt1), vreinterpretq_s64_s32(_wt3));
								int64x2_t _dt3 = vtrn2q_s64(vreinterpretq_s64_s32(_wt1), vreinterpretq_s64_s32(_wt3));
								_d0 = vreinterpretq_s32_s64(_dt0);
								_d1 = vreinterpretq_s32_s64(_dt1);
								_d2 = vreinterpretq_s32_s64(_dt2);
								_d3 = vreinterpretq_s32_s64(_dt3);
								_d4 = vtrn1q_s32(_w02n, _w13n);
								_d5 = vtrn2q_s32(_w02n, _w13n);
#else
								asm volatile(
									"vtrn.32    %q[_w0], %q[_w1]        \n"
									"vtrn.32    %q[_w2], %q[_w3]        \n"
									"vswp       %f[_w0], %e[_w2]        \n"
									"vswp       %f[_w1], %e[_w3]        \n"
									"vtrn.32    %q[_w02n], %q[_w13n]    \n"
									: [_w0] "+w"(_w0),
									[_w1] "+w"(_t1s2),
									[_w2] "+w"(_t1p2),
									[_w3] "+w"(_w3),
									[_w02n] "+w"(_w02n),
									[_w13n] "+w"(_w13n)
									:
									: "cc", "memory");
								_d0 = _w0;
								_d1 = _t1s2;
								_d2 = _t1p2;
								_d3 = _w3;
								_d4 = _w02n;
								_d5 = _w13n;
#endif
								// Y = A_T * w_t
								_t1p2 = vaddq_s32(_d1, _d2);
								_t3p4 = vaddq_s32(_d3, _d4);
								_t1s2 = vsubq_s32(_d1, _d2);
								_t3s4 = vsubq_s32(_d3, _d4);

								_o0 = vaddq_s32(_d0, _t1p2);
								_o0 = vaddq_s32(_o0, _t3p4);

								// _o2
								_t1p2 = vmlaq_lane_s32(_t1p2, _t3p4, _tp0, 1);

								_o3 = vaddq_s32(_d5, _t1s2);
								_o3 = vmlaq_lane_s32(_o3, _t3s4, _tp1, 1);

								// _o1
								_t1s2 = vmlaq_lane_s32(_t1s2, _t3s4, _tp1, 0);

								// save to top blob tm
								float32x4_t _ot0 = vcvtq_f32_s32(_o0);
								float32x4_t _ot1 = vcvtq_f32_s32(_t1s2);
								float32x4_t _ot2 = vcvtq_f32_s32(_t1p2);
								float32x4_t _ot3 = vcvtq_f32_s32(_o3);

								_ot0 = vmulq_n_f32(_ot0, 0.0017361112);
								_ot1 = vmulq_n_f32(_ot1, 0.0017361112);
								_ot2 = vmulq_n_f32(_ot2, 0.0017361112);
								_ot3 = vmulq_n_f32(_ot3, 0.0017361112);

								_o0 = vcvtq_s32_f32(_ot0);
								_o1 = vcvtq_s32_f32(_ot1);
								_o2 = vcvtq_s32_f32(_ot2);
								_o3 = vcvtq_s32_f32(_ot3);

								vst1q_s32(outRow0, _o0);
								vst1q_s32(outRow1, _o1);
								vst1q_s32(outRow2, _o2);
								vst1q_s32(outRow3, _o3);
#else
								int s0[6], s1[6], s2[6], s3[6], s4[6], s5[6];
								int w0[6], w1[6], w2[6], w3[6];
								int d0[4], d1[4], d2[4], d3[4], d4[4], d5[4];
								int o0[4], o1[4], o2[4], o3[4];

								// load
								for (int n = 0; n < 6; n++)
								{
									s0[n] = out_tile[n];
									s1[n] = out_tile[n + 6];
									s2[n] = out_tile[n + 12];
									s3[n] = out_tile[n + 18];
									s4[n] = out_tile[n + 24];
									s5[n] = out_tile[n + 30];
								}
								// w = A_T * W
								for (int n = 0; n < 5; n++)
								{
									w0[n] = s0[n] + s1[n] + s2[n] + s3[n] + s4[n];
									w1[n] = s1[n] - s2[n] + 2 * s3[n] - 2 * s4[n];
									w2[n] = s1[n] + s2[n] + 4 * s3[n] + 4 * s4[n];
									w3[n] = s1[n] - s2[n] + 8 * s3[n] - 8 * s4[n] + 4 * s5[n];
								}
								for (int n = 5; n < 6; n++)
								{
									w0[n] = 4 * (s0[n] + s1[n] + s2[n] + s3[n] + s4[n]);
									w1[n] = 4 * (s1[n] - s2[n] + 2 * s3[n] - 2 * s4[n]);
									w2[n] = 4 * (s1[n] + s2[n] + 4 * s3[n] + 4 * s4[n]);
									w3[n] = 4 * (s1[n] - s2[n] + 8 * s3[n] - 8 * s4[n] + 4 * s5[n]);
								}
								// transpose w to w_t
								{
									d0[0] = w0[0];
									d0[1] = w1[0];
									d0[2] = w2[0];
									d0[3] = w3[0];
									d1[0] = w0[1];
									d1[1] = w1[1];
									d1[2] = w2[1];
									d1[3] = w3[1];
									d2[0] = w0[2];
									d2[1] = w1[2];
									d2[2] = w2[2];
									d2[3] = w3[2];
									d3[0] = w0[3];
									d3[1] = w1[3];
									d3[2] = w2[3];
									d3[3] = w3[3];
									d4[0] = w0[4];
									d4[1] = w1[4];
									d4[2] = w2[4];
									d4[3] = w3[4];
									d5[0] = w0[5];
									d5[1] = w1[5];
									d5[2] = w2[5];
									d5[3] = w3[5];
								}
								// Y = A_T * w_t
								for (int n = 0; n < 4; n++)
								{
									o0[n] = d0[n] + d1[n] + d2[n] + d3[n] + d4[n];
									o1[n] = d1[n] - d2[n] + 2 * d3[n] - 2 * d4[n];
									o2[n] = d1[n] + d2[n] + 4 * d3[n] + 4 * d4[n];
									o3[n] = d1[n] - d2[n] + 8 * d3[n] - 8 * d4[n] + d5[n];
								}
								// save to top blob tm
								for (int n = 0; n < 4; n++)
								{
									outRow0[n] = o0[n] / 576;
									outRow1[n] = o1[n] / 576;
									outRow2[n] = o2[n] / 576;
									outRow3[n] = o3[n] / 576;
								}
#endif // __ARM_NEON
								out_tile += 36;

								outRow0 += 4;
								outRow1 += 4;
								outRow2 += 4;
								outRow3 += 4;
							}

							outRow0 += outw * 3;
							outRow1 += outw * 3;
							outRow2 += outw * 3;
							outRow3 += outw * 3;
						}
					}
				}
			}
			// END transform output

			// cut result pad
			cut_border_cpu(top_bordered, top, 0, top_bordered->height() - top->height(), 0, top_bordered->width() - top->width());
		}

		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv_im2col_sgemm_int8_neon(const std::shared_ptr<memory::tensor<int8_t>>& bottom, std::shared_ptr<memory::tensor<int>>& top)
		{
			int num = bottom->num();
			int w = bottom->width();
			int h = bottom->height();
			int inch = bottom->channels();
			int bottom_cstep = w * h;

			int outw = top->width();
			int outh = top->height();
			int outch = top->channels();
			const int size = outw * outh;
			const int maxk = this->kernel_size_w_ * this->kernel_size_h_;

			const int8_t* kernel_tm_gemm_int8_data = kernel_tm_gemm_int8_->cpu_data();
			int kernel_tm_gemm_int8_cstep = kernel_tm_gemm_int8_->count(2, 4);
			
			std::shared_ptr<memory::tensor<int8_t>> bottom_im2col = std::make_shared<memory::tensor<int8_t>>(std::vector<int>{1, inch, maxk, size}, this->params_.device_, memory::NCHW);
			int8_t* bottom_im2col_data = bottom_im2col->mutable_cpu_data();
			const int gap = w * this->stride_h_ - outw * this->stride_w_;
			for (size_t n = 0; n < num; n++)
			{
				const int8_t *bottom_data = bottom->cpu_data() + n * bottom_cstep * inch;
				int* top_data = top->mutable_cpu_data() + n * outw * outh * outch;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
				for (int p = 0; p < inch; p++)
				{
					const int8_t* img = bottom_data + p * bottom_cstep;
					int8_t* ptr = bottom_im2col_data + (p)*maxk * size;

					for (int u = 0; u < this->kernel_size_h_; u++)
					{
						for (int v = 0; v < this->kernel_size_w_; v++)
						{
							const signed char* sptr = img + this->dilation_h_ * u + this->dilation_w_ * v;

							for (int i = 0; i < outh; i++)
							{
								int j = 0;
								for (; j + 3 < outw; j += 4)
								{
									ptr[0] = sptr[0];
									ptr[1] = sptr[this->stride_w_];
									ptr[2] = sptr[this->stride_w_ * 2];
									ptr[3] = sptr[this->stride_w_ * 3];

									sptr += this->stride_w_ * 4;
									ptr += 4;
								}
								for (; j + 1 < outw; j += 2)
								{
									ptr[0] = sptr[0];
									ptr[1] = sptr[this->stride_w_];

									sptr += this->stride_w_ * 2;
									ptr += 2;
								}
								for (; j < outw; j++)
								{
									ptr[0] = sptr[0];

									sptr += this->stride_w_;
									ptr += 1;
								}

								sptr += gap;
							}
						}
					}
				}

				im2col_sgemm_int8_neon(kernel_tm_gemm_int8_data, kernel_tm_gemm_int8_cstep,
					bottom_im2col_data, size, maxk, inch,
					top_data, outw, outh, outch);
			}
		}

		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv1x1s1_sgemm_int8_neon(const std::shared_ptr<memory::tensor<int8_t>>& bottom,
			std::shared_ptr<memory::tensor<int>>& top)
		{
			int num = bottom->num();
			int w = bottom->width();
			int h = bottom->height();
			int inch = bottom->channels();
			int size = w * h;

			const int8_t* kernel_tm_gemm_int8_data = kernel_tm_gemm_int8_->cpu_data();
			int kernel_tm_gemm_int8_cstep = kernel_tm_gemm_int8_->count(2, 4);
			int maxk = this->kernel_size_h_ * this->kernel_size_w_;
			int outw = top->width();
			int outh = top->height();
			int outch = top->channels();
			for (size_t n = 0; n < num; n++)
			{
				const int8_t* bottom_im2col_data = bottom->cpu_data() + n * size * inch;
				int* top_data = top->mutable_cpu_data() + n * outw * outh * outch;
				im2col_sgemm_int8_neon(kernel_tm_gemm_int8_data, kernel_tm_gemm_int8_cstep,
					bottom_im2col_data, size, maxk, inch,
					top_data, outw, outh, outch);
			}
		}

		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv1x1s2_int8_neon(const std::shared_ptr<memory::tensor<int8_t>>& bottom,
			std::shared_ptr<memory::tensor<int>>& top)
		{
			int num = bottom->num();
			int w = bottom->width();
			int h = bottom->height();
			int inch = bottom->channels();
			int outw = top->width();
			int outh = top->height();
			int tailstep = w - 2 * outw + w;

			std::shared_ptr<memory::tensor<int8_t>> bottom_im2col(new memory::tensor<int8_t>(std::vector<int>{num, inch, outh, outw}, -1, memory::NCHW));
			for (size_t n = 0; n < num; n++)
			{
				const int8_t* bottom_data = bottom->cpu_data() + n * w * h * inch;
				int8_t* bottom_im2col_data = bottom_im2col->mutable_cpu_data() + n * outw * outh * inch;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
				for (int p = 0; p < inch; p++)
				{
					const signed char* r0 = bottom_data + (p)*w * h;
					signed char* outptr = bottom_im2col_data + (p)*outh * outw;

					for (int i = 0; i < outh; i++)
					{
						int j = 0;
						for (; j + 3 < outw; j += 4)
						{
							outptr[0] = r0[0];
							outptr[1] = r0[2];
							outptr[2] = r0[4];
							outptr[3] = r0[6];

							r0 += 8;
							outptr += 4;
						}
						for (; j + 1 < outw; j += 2)
						{
							outptr[0] = r0[0];
							outptr[1] = r0[2];

							r0 += 4;
							outptr += 2;
						}
						for (; j < outw; j++)
						{
							outptr[0] = r0[0];

							r0 += 2;
							outptr += 1;
						}

						r0 += tailstep;
					}
				}

				conv1x1s1_sgemm_int8_neon(bottom_im2col, top);
			}
		}

		template<typename Dtype>
		void operation_convolution_arm<Dtype>::im2col_sgemm_int8_neon(const int8_t* kernel_tm_gemm_int8_data, int kernel_tm_gemm_int8_cstep,
			const int8_t* bottom_im2col_data, int size, int maxk, int inch,
			int* top_data, int outw, int outh, int outch)
		{
			// permute
			memory::tensor<int8_t> tmp;
#if __ARM_NEON
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
			if (inch >= 8)
			{
				if (size >= 16)
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size / 16 + (size % 16) / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, inch / 8 + (inch % 8) / 4 + inch % 4, 16 * maxk}, this->params_.device_, memory::NCHW);
				else if (size >= 8)
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, inch / 8 + (inch % 8) / 4 + inch % 4, 8 * maxk}, this->params_.device_, memory::NCHW);
				else if (size >= 4)
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size / 4 + (size % 4) / 2 + size % 2, inch / 8 + (inch % 8) / 4 + inch % 4, 4 * maxk}, this->params_.device_, memory::NCHW);
				else if (size >= 2)
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size / 2 + size % 2, inch / 8 + (inch % 8) / 4 + inch % 4, 2 * maxk}, this->params_.device_, memory::NCHW);
				else
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size, inch / 8 + (inch % 8) / 4 + inch % 4, maxk}, this->params_.device_, memory::NCHW);
			}
			else if (inch >= 4)
			{
				if (size >= 16)
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size / 16 + (size % 16) / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, inch / 4 + inch % 4, 16 * maxk}, this->params_.device_, memory::NCHW);
				else if (size >= 8)
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, inch / 4 + inch % 4, 8 * maxk}, this->params_.device_, memory::NCHW);
				else if (size >= 4)
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size / 4 + (size % 4) / 2 + size % 2, inch / 4 + inch % 4, 4 * maxk}, this->params_.device_, memory::NCHW);
				else if (size >= 2)
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size / 2 + size % 2, inch / 4 + inch % 4, 2 * maxk}, this->params_.device_, memory::NCHW);
				else
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size, inch / 4 + inch % 4, maxk}, this->params_.device_, memory::NCHW);
			}
			else
			{
				if (size >= 16)
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size / 16 + (size % 16) / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, inch, 16 * maxk}, this->params_.device_, memory::NCHW);
				else if (size >= 8)
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, inch, 8 * maxk}, this->params_.device_, memory::NCHW);
				else if (size >= 4)
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size / 4 + (size % 4) / 2 + size % 2, inch, 4 * maxk}, this->params_.device_, memory::NCHW);
				else if (size >= 2)
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size / 2 + size % 2, inch, 2 * maxk}, this->params_.device_, memory::NCHW);
				else
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size, inch, maxk}, this->params_.device_, memory::NCHW);
			}
#else  // __ARM_FEATURE_DOTPROD
			if (inch >= 8)
			{
				if (size >= 4)
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size / 4 + (size % 4) / 2 + size % 2, inch / 8 + (inch % 8) / 4 + inch % 4, 4 * maxk}, this->params_.device_, memory::NCHW);
				else if (size >= 2)
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size / 2 + size % 2, inch / 8 + (inch % 8) / 4 + inch % 4, 2 * maxk}, this->params_.device_, memory::NCHW);
				else
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size, inch / 8 + (inch % 8) / 4 + inch % 4, maxk}, this->params_.device_, memory::NCHW);
			}
			else if (inch >= 4)
			{
				if (size >= 4)
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size / 4 + (size % 4) / 2 + size % 2, inch / 4 + inch % 4, 4 * maxk}, this->params_.device_, memory::NCHW);
				else if (size >= 2)
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size / 2 + size % 2, inch / 4 + inch % 4, 2 * maxk}, this->params_.device_, memory::NCHW);
				else
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size, inch / 4 + inch % 4, maxk}, this->params_.device_, memory::NCHW);
			}
			else
			{
				if (size >= 4)
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size / 4 + (size % 4) / 2 + size % 2, inch, 4 * maxk}, this->params_.device_, memory::NCHW);
				else if (size >= 2)
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size / 2 + size % 2, inch, 2 * maxk}, this->params_.device_, memory::NCHW);
				else
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size, inch, maxk}, this->params_.device_, memory::NCHW);
			}
#endif // __ARM_FEATURE_DOTPROD
#else  // __aarch64__
			if (inch >= 8)
			{
				if (size >= 2)
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size / 2 + size % 2, inch / 8 + (inch % 8) / 4 + inch % 4, 2 * maxk}, this->params_.device_, memory::NCHW);
				else
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size, inch / 8 + (inch % 8) / 4 + inch % 4, maxk}, this->params_.device_, memory::NCHW);
			}
			else if (inch >= 4)
			{
				if (size >= 2)
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size / 2 + size % 2, inch / 4 + inch % 4, 2 * maxk}, this->params_.device_, memory::NCHW);
				else
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size, inch / 4 + inch % 4, maxk}, this->params_.device_, memory::NCHW);
			}
			else
			{
				if (size >= 2)
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size / 2 + size % 2, inch, 2 * maxk}, this->params_.device_, memory::NCHW);
				else
					tmp = memory::tensor<int8_t>(std::vector<int>{1, size, inch, maxk}, this->params_.device_, memory::NCHW);
			}
#endif // __aarch64__
			signed char* tmp_data = tmp.mutable_cpu_data();
			int tmp_cstep = tmp.count(2, 4);
			{
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
				int nn_size = size >> 4;
				int remain_size_start = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
				for (int ii = 0; ii < nn_size; ii++)
				{
					int i = remain_size_start + ii * 16;

					signed char* tmpptr = tmp_data + tmp_cstep * (i / 16);

					int q = 0;
					for (; q + 7 < inch; q += 8)
					{
						const signed char* img0 = bottom_im2col_data + (q)*maxk * size + i;
						const signed char* img1 = bottom_im2col_data + (q + 1) * maxk * size + i;
						const signed char* img2 = bottom_im2col_data + (q + 2) * maxk * size + i;
						const signed char* img3 = bottom_im2col_data + (q + 3) * maxk * size + i;
						const signed char* img4 = bottom_im2col_data + (q + 4) * maxk * size + i;
						const signed char* img5 = bottom_im2col_data + (q + 5) * maxk * size + i;
						const signed char* img6 = bottom_im2col_data + (q + 6) * maxk * size + i;
						const signed char* img7 = bottom_im2col_data + (q + 7) * maxk * size + i;

						for (int k = 0; k < maxk; k++)
						{
							asm volatile(
								"ld1    {v0.16b}, [%0]              \n"
								"ld1    {v1.16b}, [%1]              \n"
								"ld1    {v2.16b}, [%2]              \n"
								"ld1    {v3.16b}, [%3]              \n"
								"ld1    {v4.16b}, [%4]              \n"
								"ld1    {v5.16b}, [%5]              \n"
								"ld1    {v6.16b}, [%6]              \n"
								"ld1    {v7.16b}, [%7]              \n"
								"st4    {v0.16b, v1.16b, v2.16b, v3.16b}, [%8], #64 \n"
								"st4    {v4.16b, v5.16b, v6.16b, v7.16b}, [%8], #64 \n"
								: "=r"(img0), // %0
								"=r"(img1),
								"=r"(img2),
								"=r"(img3),
								"=r"(img4),
								"=r"(img5),
								"=r"(img6),
								"=r"(img7),
								"=r"(tmpptr) // %8
								: "0"(img0),
								"1"(img1),
								"2"(img2),
								"3"(img3),
								"4"(img4),
								"5"(img5),
								"6"(img6),
								"7"(img7),
								"8"(tmpptr)
								: "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
							img0 += size;
							img1 += size;
							img2 += size;
							img3 += size;
							img4 += size;
							img5 += size;
							img6 += size;
							img7 += size;
						}
					}
					for (; q + 3 < inch; q += 4)
					{
						const signed char* img0 = bottom_im2col_data + (q)*maxk * size + i;
						const signed char* img1 = bottom_im2col_data + (q + 1) * maxk * size + i;
						const signed char* img2 = bottom_im2col_data + (q + 2) * maxk * size + i;
						const signed char* img3 = bottom_im2col_data + (q + 3) * maxk * size + i;

						for (int k = 0; k < maxk; k++)
						{
							asm volatile(
								"ld1    {v0.16b}, [%0]              \n"
								"ld1    {v1.16b}, [%1]              \n"
								"ld1    {v2.16b}, [%2]              \n"
								"ld1    {v3.16b}, [%3]              \n"
								"st4    {v0.16b, v1.16b, v2.16b, v3.16b}, [%4], #64 \n"
								: "=r"(img0), // %0
								"=r"(img1),
								"=r"(img2),
								"=r"(img3),
								"=r"(tmpptr) // %4
								: "0"(img0),
								"1"(img1),
								"2"(img2),
								"3"(img3),
								"4"(tmpptr)
								: "memory", "v0", "v1", "v2", "v3");
							img0 += size;
							img1 += size;
							img2 += size;
							img3 += size;
						}
					}
					for (; q < inch; q++)
					{
						const signed char* img0 = bottom_im2col_data + (q)*maxk * size + i;

						for (int k = 0; k < maxk; k++)
						{
							asm volatile(
								"prfm   pldl1keep, [%0, #128]   \n"
								"ld1    {v0.16b}, [%0]          \n"
								"st1    {v0.16b}, [%1], #16     \n"
								: "=r"(img0),  // %0
								"=r"(tmpptr) // %1
								: "0"(img0),
								"1"(tmpptr)
								: "memory", "v0");
							img0 += size;
						}
					}
				}

				remain_size_start += nn_size << 4;
				nn_size = (size - remain_size_start) >> 3;
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
				for (int ii = 0; ii < nn_size; ii++)
				{
					int i = remain_size_start + ii * 8;

					signed char* tmpptr = tmp_data + (i / 16 + (i % 16) / 8) * tmp_cstep;

					int q = 0;
					for (; q + 7 < inch; q += 8)
					{
						const signed char* img0 = bottom_im2col_data + (q)*maxk * size + i;
						const signed char* img1 = bottom_im2col_data + (q + 1) * maxk * size + i;
						const signed char* img2 = bottom_im2col_data + (q + 2) * maxk * size + i;
						const signed char* img3 = bottom_im2col_data + (q + 3) * maxk * size + i;
						const signed char* img4 = bottom_im2col_data + (q + 4) * maxk * size + i;
						const signed char* img5 = bottom_im2col_data + (q + 5) * maxk * size + i;
						const signed char* img6 = bottom_im2col_data + (q + 6) * maxk * size + i;
						const signed char* img7 = bottom_im2col_data + (q + 7) * maxk * size + i;

						for (int k = 0; k < maxk; k++)
						{
							asm volatile(
								"ld1    {v0.8b}, [%0]               \n"
								"ld1    {v1.8b}, [%1]               \n"
								"ld1    {v2.8b}, [%2]               \n"
								"ld1    {v3.8b}, [%3]               \n"
								"ld1    {v4.8b}, [%4]               \n"
								"ld1    {v5.8b}, [%5]               \n"
								"ld1    {v6.8b}, [%6]               \n"
								"ld1    {v7.8b}, [%7]               \n"
								"st4    {v0.8b, v1.8b, v2.8b, v3.8b}, [%8], #32 \n"
								"st4    {v4.8b, v5.8b, v6.8b, v7.8b}, [%8], #32 \n"
								: "=r"(img0), // %0
								"=r"(img1),
								"=r"(img2),
								"=r"(img3),
								"=r"(img4),
								"=r"(img5),
								"=r"(img6),
								"=r"(img7),
								"=r"(tmpptr) // %8
								: "0"(img0),
								"1"(img1),
								"2"(img2),
								"3"(img3),
								"4"(img4),
								"5"(img5),
								"6"(img6),
								"7"(img7),
								"8"(tmpptr)
								: "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
							img0 += size;
							img1 += size;
							img2 += size;
							img3 += size;
							img4 += size;
							img5 += size;
							img6 += size;
							img7 += size;
						}
					}
					for (; q + 3 < inch; q += 4)
					{
						const signed char* img0 = bottom_im2col_data + (q)*maxk * size + i;
						const signed char* img1 = bottom_im2col_data + (q + 1) * maxk * size + i;
						const signed char* img2 = bottom_im2col_data + (q + 2) * maxk * size + i;
						const signed char* img3 = bottom_im2col_data + (q + 3) * maxk * size + i;

						for (int k = 0; k < maxk; k++)
						{
							asm volatile(
								"ld1    {v0.8b}, [%0]               \n"
								"ld1    {v1.8b}, [%1]               \n"
								"ld1    {v2.8b}, [%2]               \n"
								"ld1    {v3.8b}, [%3]               \n"
								"st4    {v0.8b, v1.8b, v2.8b, v3.8b}, [%4], #32 \n"
								: "=r"(img0), // %0
								"=r"(img1),
								"=r"(img2),
								"=r"(img3),
								"=r"(tmpptr) // %4
								: "0"(img0),
								"1"(img1),
								"2"(img2),
								"3"(img3),
								"4"(tmpptr)
								: "memory", "v0", "v1", "v2", "v3");
							img0 += size;
							img1 += size;
							img2 += size;
							img3 += size;
						}
					}
					for (; q < inch; q++)
					{
						const signed char* img0 = bottom_im2col_data + (q)*maxk * size + i;

						for (int k = 0; k < maxk; k++)
						{
							asm volatile(
								"prfm   pldl1keep, [%0, #64]    \n"
								"ld1    {v0.8b}, [%0]           \n"
								"st1    {v0.8b}, [%1], #8       \n"
								: "=r"(img0),  // %0
								"=r"(tmpptr) // %1
								: "0"(img0),
								"1"(tmpptr)
								: "memory", "v0");
							img0 += size;
						}
					}
				}

				remain_size_start += nn_size << 3;
				nn_size = (size - remain_size_start) >> 2;
#else  // __ARM_FEATURE_DOTPROD
				int remain_size_start = 0;
				int nn_size = (size - remain_size_start) >> 2;
#endif // __ARM_FEATURE_DOTPROD

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
				for (int ii = 0; ii < nn_size; ii++)
				{
					int i = remain_size_start + ii * 4;

#if __ARM_FEATURE_DOTPROD
					signed char* tmpptr = tmp_data + (i / 16 + (i % 16) / 8 + (i % 8) / 4) * tmp_cstep;
#else
					signed char* tmpptr = tmp_data + (i / 4) * tmp_cstep;
#endif

					int q = 0;
					for (; q + 7 < inch; q += 8)
					{
						const signed char* img0 = bottom_im2col_data + (q)*maxk * size + i;
						const signed char* img1 = bottom_im2col_data + (q + 1) * maxk * size + i;
						const signed char* img2 = bottom_im2col_data + (q + 2) * maxk * size + i;
						const signed char* img3 = bottom_im2col_data + (q + 3) * maxk * size + i;
						const signed char* img4 = bottom_im2col_data + (q + 4) * maxk * size + i;
						const signed char* img5 = bottom_im2col_data + (q + 5) * maxk * size + i;
						const signed char* img6 = bottom_im2col_data + (q + 6) * maxk * size + i;
						const signed char* img7 = bottom_im2col_data + (q + 7) * maxk * size + i;

						for (int k = 0; k < maxk; k++)
						{
#if __ARM_FEATURE_DOTPROD
							tmpptr[0] = img0[0];
							tmpptr[1] = img1[0];
							tmpptr[2] = img2[0];
							tmpptr[3] = img3[0];
							tmpptr[4] = img0[1];
							tmpptr[5] = img1[1];
							tmpptr[6] = img2[1];
							tmpptr[7] = img3[1];
							tmpptr += 8;

							tmpptr[0] = img0[2];
							tmpptr[1] = img1[2];
							tmpptr[2] = img2[2];
							tmpptr[3] = img3[2];
							tmpptr[4] = img0[3];
							tmpptr[5] = img1[3];
							tmpptr[6] = img2[3];
							tmpptr[7] = img3[3];
							tmpptr += 8;

							tmpptr[0] = img4[0];
							tmpptr[1] = img5[0];
							tmpptr[2] = img6[0];
							tmpptr[3] = img7[0];
							tmpptr[4] = img4[1];
							tmpptr[5] = img5[1];
							tmpptr[6] = img6[1];
							tmpptr[7] = img7[1];
							tmpptr += 8;

							tmpptr[0] = img4[2];
							tmpptr[1] = img5[2];
							tmpptr[2] = img6[2];
							tmpptr[3] = img7[2];
							tmpptr[4] = img4[3];
							tmpptr[5] = img5[3];
							tmpptr[6] = img6[3];
							tmpptr[7] = img7[3];
							tmpptr += 8;
#else
							tmpptr[0] = img0[0];
							tmpptr[1] = img1[0];
							tmpptr[2] = img2[0];
							tmpptr[3] = img3[0];
							tmpptr[4] = img4[0];
							tmpptr[5] = img5[0];
							tmpptr[6] = img6[0];
							tmpptr[7] = img7[0];
							tmpptr += 8;

							tmpptr[0] = img0[1];
							tmpptr[1] = img1[1];
							tmpptr[2] = img2[1];
							tmpptr[3] = img3[1];
							tmpptr[4] = img4[1];
							tmpptr[5] = img5[1];
							tmpptr[6] = img6[1];
							tmpptr[7] = img7[1];
							tmpptr += 8;

							tmpptr[0] = img0[2];
							tmpptr[1] = img1[2];
							tmpptr[2] = img2[2];
							tmpptr[3] = img3[2];
							tmpptr[4] = img4[2];
							tmpptr[5] = img5[2];
							tmpptr[6] = img6[2];
							tmpptr[7] = img7[2];
							tmpptr += 8;

							tmpptr[0] = img0[3];
							tmpptr[1] = img1[3];
							tmpptr[2] = img2[3];
							tmpptr[3] = img3[3];
							tmpptr[4] = img4[3];
							tmpptr[5] = img5[3];
							tmpptr[6] = img6[3];
							tmpptr[7] = img7[3];
							tmpptr += 8;
#endif // __ARM_FEATURE_DOTPROD

							img0 += size;
							img1 += size;
							img2 += size;
							img3 += size;
							img4 += size;
							img5 += size;
							img6 += size;
							img7 += size;
						}
					}
					for (; q + 3 < inch; q += 4)
					{
						const signed char* img0 = bottom_im2col_data + (q)*maxk * size + i;
						const signed char* img1 = bottom_im2col_data + (q + 1) * maxk * size + i;
						const signed char* img2 = bottom_im2col_data + (q + 2) * maxk * size + i;
						const signed char* img3 = bottom_im2col_data + (q + 3) * maxk * size + i;

						for (int k = 0; k < maxk; k++)
						{
							tmpptr[0] = img0[0];
							tmpptr[1] = img1[0];
							tmpptr[2] = img2[0];
							tmpptr[3] = img3[0];
							tmpptr[4] = img0[1];
							tmpptr[5] = img1[1];
							tmpptr[6] = img2[1];
							tmpptr[7] = img3[1];
							tmpptr += 8;

							tmpptr[0] = img0[2];
							tmpptr[1] = img1[2];
							tmpptr[2] = img2[2];
							tmpptr[3] = img3[2];
							tmpptr[4] = img0[3];
							tmpptr[5] = img1[3];
							tmpptr[6] = img2[3];
							tmpptr[7] = img3[3];
							tmpptr += 8;

							img0 += size;
							img1 += size;
							img2 += size;
							img3 += size;
						}
					}
					for (; q < inch; q++)
					{
						const signed char* img0 = bottom_im2col_data + (q)*maxk * size + i;

						for (int k = 0; k < maxk; k++)
						{
							tmpptr[0] = img0[0];
							tmpptr[1] = img0[1];
							tmpptr[2] = img0[2];
							tmpptr[3] = img0[3];

							tmpptr += 4;

							img0 += size;
						}
					}
				}

				remain_size_start += nn_size << 2;
				nn_size = (size - remain_size_start) >> 1;
#else
				int remain_size_start = 0;
				int nn_size = (size - remain_size_start) >> 1;
#endif

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
				for (int ii = 0; ii < nn_size; ii++)
				{
					int i = remain_size_start + ii * 2;

#if __aarch64__
#if __ARM_FEATURE_DOTPROD
					signed char* tmpptr = tmp_data + (i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2) * tmp_cstep;
#else
					signed char* tmpptr = tmp_data + (i / 4 + (i % 4) / 2) * tmp_cstep;
#endif
#else
					signed char* tmpptr = tmp_data + (i / 2) * tmp_cstep;
#endif

					int q = 0;
					for (; q + 7 < inch; q += 8)
					{
						const signed char* img0 = bottom_im2col_data + (q)*maxk * size + i;
						const signed char* img1 = bottom_im2col_data + (q + 1) * maxk * size + i;
						const signed char* img2 = bottom_im2col_data + (q + 2) * maxk * size + i;
						const signed char* img3 = bottom_im2col_data + (q + 3) * maxk * size + i;
						const signed char* img4 = bottom_im2col_data + (q + 4) * maxk * size + i;
						const signed char* img5 = bottom_im2col_data + (q + 5) * maxk * size + i;
						const signed char* img6 = bottom_im2col_data + (q + 6) * maxk * size + i;
						const signed char* img7 = bottom_im2col_data + (q + 7) * maxk * size + i;

						for (int k = 0; k < maxk; k++)
						{
#if __ARM_FEATURE_DOTPROD
							tmpptr[0] = img0[0];
							tmpptr[1] = img1[0];
							tmpptr[2] = img2[0];
							tmpptr[3] = img3[0];
							tmpptr[4] = img0[1];
							tmpptr[5] = img1[1];
							tmpptr[6] = img2[1];
							tmpptr[7] = img3[1];
							tmpptr += 8;

							tmpptr[0] = img4[0];
							tmpptr[1] = img5[0];
							tmpptr[2] = img6[0];
							tmpptr[3] = img7[0];
							tmpptr[4] = img4[1];
							tmpptr[5] = img5[1];
							tmpptr[6] = img6[1];
							tmpptr[7] = img7[1];
							tmpptr += 8;
#else
							tmpptr[0] = img0[0];
							tmpptr[1] = img1[0];
							tmpptr[2] = img2[0];
							tmpptr[3] = img3[0];
							tmpptr[4] = img4[0];
							tmpptr[5] = img5[0];
							tmpptr[6] = img6[0];
							tmpptr[7] = img7[0];
							tmpptr += 8;

							tmpptr[0] = img0[1];
							tmpptr[1] = img1[1];
							tmpptr[2] = img2[1];
							tmpptr[3] = img3[1];
							tmpptr[4] = img4[1];
							tmpptr[5] = img5[1];
							tmpptr[6] = img6[1];
							tmpptr[7] = img7[1];
							tmpptr += 8;
#endif // __ARM_FEATURE_DOTPROD

							img0 += size;
							img1 += size;
							img2 += size;
							img3 += size;
							img4 += size;
							img5 += size;
							img6 += size;
							img7 += size;
						}
					}
					for (; q + 3 < inch; q += 4)
					{
						const signed char* img0 = bottom_im2col_data + (q)*maxk * size + i;
						const signed char* img1 = bottom_im2col_data + (q + 1) * maxk * size + i;
						const signed char* img2 = bottom_im2col_data + (q + 2) * maxk * size + i;
						const signed char* img3 = bottom_im2col_data + (q + 3) * maxk * size + i;

						for (int k = 0; k < maxk; k++)
						{
							tmpptr[0] = img0[0];
							tmpptr[1] = img1[0];
							tmpptr[2] = img2[0];
							tmpptr[3] = img3[0];
							tmpptr[4] = img0[1];
							tmpptr[5] = img1[1];
							tmpptr[6] = img2[1];
							tmpptr[7] = img3[1];
							tmpptr += 8;

							img0 += size;
							img1 += size;
							img2 += size;
							img3 += size;
						}
					}
					for (; q < inch; q++)
					{
						const signed char* img0 = bottom_im2col_data + (q)*maxk * size + i;

						for (int k = 0; k < maxk; k++)
						{
							tmpptr[0] = img0[0];
							tmpptr[1] = img0[1];

							tmpptr += 2;

							img0 += size;
						}
					}
				}

				remain_size_start += nn_size << 1;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
				for (int i = remain_size_start; i < size; i++)
				{
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
					signed char* tmpptr = tmp_data + (i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2) * tmp_cstep;
#else
					signed char* tmpptr = tmp_data + (i / 4 + (i % 4) / 2 + i % 2) * tmp_cstep;
#endif
#else
					signed char* tmpptr = tmp_data + (i / 2 + i % 2) * tmp_cstep;
#endif

					int q = 0;
					for (; q + 7 < inch; q += 8)
					{
						const signed char* img0 = bottom_im2col_data + (q)*maxk * size + i;
						const signed char* img1 = bottom_im2col_data + (q + 1) * maxk * size + i;
						const signed char* img2 = bottom_im2col_data + (q + 2) * maxk * size + i;
						const signed char* img3 = bottom_im2col_data + (q + 3) * maxk * size + i;
						const signed char* img4 = bottom_im2col_data + (q + 4) * maxk * size + i;
						const signed char* img5 = bottom_im2col_data + (q + 5) * maxk * size + i;
						const signed char* img6 = bottom_im2col_data + (q + 6) * maxk * size + i;
						const signed char* img7 = bottom_im2col_data + (q + 7) * maxk * size + i;

						for (int k = 0; k < maxk; k++)
						{
							tmpptr[0] = img0[0];
							tmpptr[1] = img1[0];
							tmpptr[2] = img2[0];
							tmpptr[3] = img3[0];
							tmpptr[4] = img4[0];
							tmpptr[5] = img5[0];
							tmpptr[6] = img6[0];
							tmpptr[7] = img7[0];
							tmpptr += 8;

							img0 += size;
							img1 += size;
							img2 += size;
							img3 += size;
							img4 += size;
							img5 += size;
							img6 += size;
							img7 += size;
						}
					}
					for (; q + 3 < inch; q += 4)
					{
						const signed char* img0 = bottom_im2col_data + (q)*maxk * size + i;
						const signed char* img1 = bottom_im2col_data + (q + 1) * maxk * size + i;
						const signed char* img2 = bottom_im2col_data + (q + 2) * maxk * size + i;
						const signed char* img3 = bottom_im2col_data + (q + 3) * maxk * size + i;

						for (int k = 0; k < maxk; k++)
						{
							tmpptr[0] = img0[0];
							tmpptr[1] = img1[0];
							tmpptr[2] = img2[0];
							tmpptr[3] = img3[0];
							tmpptr += 4;

							img0 += size;
							img1 += size;
							img2 += size;
							img3 += size;
						}
					}
					for (; q < inch; q++)
					{
						const signed char* img0 = bottom_im2col_data + (q)*maxk * size + i;

						for (int k = 0; k < maxk; k++)
						{
							tmpptr[0] = img0[0];

							tmpptr += 1;

							img0 += size;
						}
					}
				}
			}
#else // __ARM_NEON
			tmp = memory::tensor<int8_t>(std::vector<int>{1, size, inch, maxk}, this->params_.device_, memory::NCHW);
			signed char* tmp_data = tmp.mutable_cpu_data();
			int tmp_cstep = tmp.count(2, 4);
			{
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
				for (int i = 0; i < size; i++)
				{
					signed char* tmpptr = tmp_data + (i)*tmp_cstep;

					int q = 0;
					for (; q < inch; q++)
					{
						const signed char* img0 = bottom_im2col_data + (q)*maxk * size + i;

						for (int k = 0; k < maxk; k++)
						{
							tmpptr[0] = img0[0];

							tmpptr += 1;

							img0 += size;
						}
					}
				}
			}
#endif // __ARM_NEON

			int nn_outch = 0;
			int remain_outch_start = 0;

#if __ARM_NEON
			nn_outch = outch >> 2;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
			for (int pp = 0; pp < nn_outch; pp++)
			{
				int p = pp * 4;

				int* outptr0 = top_data + (p)*outw * outh;
				int* outptr1 = top_data + (p + 1) * outw * outh;
				int* outptr2 = top_data + (p + 2) * outw * outh;
				int* outptr3 = top_data + (p + 3) * outw * outh;

				int i = 0;
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
				for (; i + 15 < size; i += 16)
				{
					const signed char* tmpptr = tmp_data + (i / 16) * tmp_cstep;
					const signed char* kptr0 = kernel_tm_gemm_int8_data + (p / 4) * kernel_tm_gemm_int8_cstep;

					int nn = (inch / 8) * maxk;
					int nn4 = ((inch % 8) / 4) * maxk;
					int nn1 = (inch % 4) * maxk;

					asm volatile(
						"eor    v16.16b, v16.16b, v16.16b   \n"
						"eor    v17.16b, v17.16b, v17.16b   \n"
						"eor    v18.16b, v18.16b, v18.16b   \n"
						"eor    v19.16b, v19.16b, v19.16b   \n"
						"eor    v20.16b, v20.16b, v20.16b   \n"
						"eor    v21.16b, v21.16b, v21.16b   \n"
						"eor    v22.16b, v22.16b, v22.16b   \n"
						"eor    v23.16b, v23.16b, v23.16b   \n"
						"eor    v24.16b, v24.16b, v24.16b   \n"
						"eor    v25.16b, v25.16b, v25.16b   \n"
						"eor    v26.16b, v26.16b, v26.16b   \n"
						"eor    v27.16b, v27.16b, v27.16b   \n"
						"eor    v28.16b, v28.16b, v28.16b   \n"
						"eor    v29.16b, v29.16b, v29.16b   \n"
						"eor    v30.16b, v30.16b, v30.16b   \n"
						"eor    v31.16b, v31.16b, v31.16b   \n"

						"cmp    %w4, #0                     \n"
						"beq    1f                          \n"

						"ld1    {v8.16b}, [%8], #16         \n" // _w0123_l

						"ld1    {v0.16b}, [%7], #16         \n" // _val0123_l

						"0:                                 \n"

						"ld1    {v1.16b}, [%7], #16         \n" // _val4567_l

						"sdot   v16.4s, v8.16b, v0.4b[0]    \n"
						"sdot   v17.4s, v8.16b, v0.4b[1]    \n"
						"sdot   v18.4s, v8.16b, v0.4b[2]    \n"
						"sdot   v19.4s, v8.16b, v0.4b[3]    \n"

						"ld1    {v2.16b}, [%7], #16         \n" // _val891011_l

						"sdot   v20.4s, v8.16b, v1.4b[0]    \n"
						"sdot   v21.4s, v8.16b, v1.4b[1]    \n"
						"sdot   v22.4s, v8.16b, v1.4b[2]    \n"
						"sdot   v23.4s, v8.16b, v1.4b[3]    \n"

						"ld1    {v3.16b}, [%7], #16         \n" // _val12131415_l

						"sdot   v24.4s, v8.16b, v2.4b[0]    \n"
						"sdot   v25.4s, v8.16b, v2.4b[1]    \n"

						"ld1    {v9.16b}, [%8], #16         \n" // _w0123_h

						"sdot   v26.4s, v8.16b, v2.4b[2]    \n"
						"sdot   v27.4s, v8.16b, v2.4b[3]    \n"

						"ld1    {v4.16b}, [%7], #16         \n" // _val0123_h

						"sdot   v28.4s, v8.16b, v3.4b[0]    \n"
						"sdot   v29.4s, v8.16b, v3.4b[1]    \n"
						"sdot   v30.4s, v8.16b, v3.4b[2]    \n"
						"sdot   v31.4s, v8.16b, v3.4b[3]    \n"

						"ld1    {v5.16b}, [%7], #16         \n" // _val4567_h

						"sdot   v16.4s, v9.16b, v4.4b[0]    \n"
						"sdot   v17.4s, v9.16b, v4.4b[1]    \n"
						"sdot   v18.4s, v9.16b, v4.4b[2]    \n"
						"sdot   v19.4s, v9.16b, v4.4b[3]    \n"

						"ld1    {v6.16b}, [%7], #16         \n" // _val891011_h

						"sdot   v20.4s, v9.16b, v5.4b[0]    \n"
						"sdot   v21.4s, v9.16b, v5.4b[1]    \n"
						"sdot   v22.4s, v9.16b, v5.4b[2]    \n"
						"sdot   v23.4s, v9.16b, v5.4b[3]    \n"

						"ld1    {v7.16b}, [%7], #16         \n" // _val12131415_h

						"sdot   v24.4s, v9.16b, v6.4b[0]    \n"
						"sdot   v25.4s, v9.16b, v6.4b[1]    \n"

						"ld1    {v8.16b}, [%8], #16         \n" // _w0123_l

						"sdot   v26.4s, v9.16b, v6.4b[2]    \n"
						"sdot   v27.4s, v9.16b, v6.4b[3]    \n"

						"ld1    {v0.16b}, [%7], #16         \n" // _val0123_l

						"sdot   v28.4s, v9.16b, v7.4b[0]    \n"
						"sdot   v29.4s, v9.16b, v7.4b[1]    \n"

						"subs   %w4, %w4, #1                \n"

						"sdot   v30.4s, v9.16b, v7.4b[2]    \n"
						"sdot   v31.4s, v9.16b, v7.4b[3]    \n"

						"bne    0b                          \n"

						"sub    %7, %7, #16                 \n"
						"sub    %8, %8, #16                 \n"

						"1:                                 \n"

						"cmp    %w5, #0                     \n"
						"beq    3f                          \n"

						"2:                                 \n"

						"ld1    {v8.16b}, [%8], #16         \n"

						"ld1    {v0.16b, v1.16b, v2.16b, v3.16b}, [%7], #64 \n"

						"sdot   v16.4s, v8.16b, v0.4b[0]    \n"
						"sdot   v17.4s, v8.16b, v0.4b[1]    \n"
						"sdot   v18.4s, v8.16b, v0.4b[2]    \n"
						"sdot   v19.4s, v8.16b, v0.4b[3]    \n"
						"sdot   v20.4s, v8.16b, v1.4b[0]    \n"
						"sdot   v21.4s, v8.16b, v1.4b[1]    \n"
						"sdot   v22.4s, v8.16b, v1.4b[2]    \n"
						"sdot   v23.4s, v8.16b, v1.4b[3]    \n"
						"sdot   v24.4s, v8.16b, v2.4b[0]    \n"
						"sdot   v25.4s, v8.16b, v2.4b[1]    \n"
						"sdot   v26.4s, v8.16b, v2.4b[2]    \n"
						"sdot   v27.4s, v8.16b, v2.4b[3]    \n"
						"sdot   v28.4s, v8.16b, v3.4b[0]    \n"
						"sdot   v29.4s, v8.16b, v3.4b[1]    \n"

						"subs   %w5, %w5, #1                \n"

						"sdot   v30.4s, v8.16b, v3.4b[2]    \n"
						"sdot   v31.4s, v8.16b, v3.4b[3]    \n"

						"bne    2b                          \n"

						"3:                                 \n"

						"lsr    w4, %w6, #2                 \n" // w4 = nn1 >> 2
						"cmp    w4, #0                      \n"
						"beq    5f                          \n"

						"4:                                 \n"

						"ld1    {v8.8b, v9.8b}, [%8], #16   \n"

						"ld4    {v0.16b, v1.16b, v2.16b, v3.16b}, [%7], #64 \n"

						"uzp1   v10.8b, v8.8b, v9.8b        \n"
						"uzp2   v11.8b, v8.8b, v9.8b        \n"

						"uzp1   v4.16b, v0.16b, v1.16b      \n"
						"uzp2   v5.16b, v0.16b, v1.16b      \n"
						"uzp1   v6.16b, v2.16b, v3.16b      \n"
						"uzp2   v7.16b, v2.16b, v3.16b      \n"

						"uzp1   v8.8b, v10.8b, v11.8b       \n"
						"uzp2   v9.8b, v10.8b, v11.8b       \n"

						"uzp1   v0.16b, v4.16b, v5.16b      \n" // 0 1 4 5
						"uzp2   v1.16b, v4.16b, v5.16b      \n" // 8 9 c d

						"mov    v8.d[1], v9.d[0]            \n" // _w

						"uzp1   v2.16b, v6.16b, v7.16b      \n" // 2 3 6 7
						"uzp2   v3.16b, v6.16b, v7.16b      \n" // a b e f

						"sdot   v16.4s, v8.16b, v0.4b[0]    \n"
						"sdot   v17.4s, v8.16b, v0.4b[1]    \n"
						"sdot   v18.4s, v8.16b, v2.4b[0]    \n"
						"sdot   v19.4s, v8.16b, v2.4b[1]    \n"
						"sdot   v20.4s, v8.16b, v0.4b[2]    \n"
						"sdot   v21.4s, v8.16b, v0.4b[3]    \n"
						"sdot   v22.4s, v8.16b, v2.4b[2]    \n"
						"sdot   v23.4s, v8.16b, v2.4b[3]    \n"
						"sdot   v24.4s, v8.16b, v1.4b[0]    \n"
						"sdot   v25.4s, v8.16b, v1.4b[1]    \n"
						"sdot   v26.4s, v8.16b, v3.4b[0]    \n"
						"sdot   v27.4s, v8.16b, v3.4b[1]    \n"
						"sdot   v28.4s, v8.16b, v1.4b[2]    \n"
						"sdot   v29.4s, v8.16b, v1.4b[3]    \n"
						"sdot   v30.4s, v8.16b, v3.4b[2]    \n"
						"sdot   v31.4s, v8.16b, v3.4b[3]    \n"

						"subs   w4, w4, #1                  \n"
						"bne    4b                          \n"

						"5:                                 \n"

						"and    w4, %w6, #3                 \n" // w4 = remain = nn1 & 3
						"cmp    w4, #0                      \n" // w4 > 0
						"beq    7f                          \n"

						"6:                                 \n"

						"ld1    {v1.8b}, [%8]               \n"
						"ld1    {v0.16b}, [%7]              \n"

						"sshll  v1.8h, v1.8b, #0            \n"
						"sshll  v2.8h, v0.8b, #0            \n"
						"sshll2 v3.8h, v0.16b, #0           \n"

						"smlal  v16.4s, v1.4h, v2.h[0]      \n"
						"smlal  v17.4s, v1.4h, v2.h[1]      \n"
						"smlal  v18.4s, v1.4h, v2.h[2]      \n"
						"smlal  v19.4s, v1.4h, v2.h[3]      \n"
						"smlal  v20.4s, v1.4h, v2.h[4]      \n"
						"smlal  v21.4s, v1.4h, v2.h[5]      \n"
						"smlal  v22.4s, v1.4h, v2.h[6]      \n"
						"smlal  v23.4s, v1.4h, v2.h[7]      \n"
						"smlal  v24.4s, v1.4h, v3.h[0]      \n"
						"smlal  v25.4s, v1.4h, v3.h[1]      \n"
						"smlal  v26.4s, v1.4h, v3.h[2]      \n"
						"smlal  v27.4s, v1.4h, v3.h[3]      \n"
						"smlal  v28.4s, v1.4h, v3.h[4]      \n"
						"smlal  v29.4s, v1.4h, v3.h[5]      \n"
						"smlal  v30.4s, v1.4h, v3.h[6]      \n"
						"smlal  v31.4s, v1.4h, v3.h[7]      \n"

						"add    %7, %7, #16                 \n"
						"add    %8, %8, #4                  \n"

						"subs   w4, w4, #1                  \n"
						"bne    6b                          \n"

						"7:                                 \n"

						// transpose 4x16
						"trn1   v0.4s, v16.4s, v17.4s       \n"
						"trn2   v1.4s, v16.4s, v17.4s       \n"
						"trn1   v2.4s, v18.4s, v19.4s       \n"
						"trn2   v3.4s, v18.4s, v19.4s       \n"
						"trn1   v4.4s, v20.4s, v21.4s       \n"
						"trn2   v5.4s, v20.4s, v21.4s       \n"
						"trn1   v6.4s, v22.4s, v23.4s       \n"
						"trn2   v7.4s, v22.4s, v23.4s       \n"
						"trn1   v8.4s, v24.4s, v25.4s       \n"
						"trn2   v9.4s, v24.4s, v25.4s       \n"
						"trn1   v10.4s, v26.4s, v27.4s      \n"
						"trn2   v11.4s, v26.4s, v27.4s      \n"
						"trn1   v12.4s, v28.4s, v29.4s      \n"
						"trn2   v13.4s, v28.4s, v29.4s      \n"
						"trn1   v14.4s, v30.4s, v31.4s      \n"
						"trn2   v15.4s, v30.4s, v31.4s      \n"

						"trn1   v16.2d, v0.2d, v2.2d        \n"
						"trn2   v24.2d, v0.2d, v2.2d        \n"
						"trn1   v20.2d, v1.2d, v3.2d        \n"
						"trn2   v28.2d, v1.2d, v3.2d        \n"

						"trn1   v17.2d, v4.2d, v6.2d        \n"
						"trn2   v25.2d, v4.2d, v6.2d        \n"
						"trn1   v21.2d, v5.2d, v7.2d        \n"
						"trn2   v29.2d, v5.2d, v7.2d        \n"

						"trn1   v18.2d, v8.2d, v10.2d       \n"
						"trn2   v26.2d, v8.2d, v10.2d       \n"
						"trn1   v22.2d, v9.2d, v11.2d       \n"
						"trn2   v30.2d, v9.2d, v11.2d       \n"

						"trn1   v19.2d, v12.2d, v14.2d      \n"
						"trn2   v27.2d, v12.2d, v14.2d      \n"
						"trn1   v23.2d, v13.2d, v15.2d      \n"
						"trn2   v31.2d, v13.2d, v15.2d      \n"

						"st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
						"st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%1], #64 \n"
						"st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%2], #64 \n"
						"st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%3], #64 \n"
						: "=r"(outptr0),
						"=r"(outptr1),
						"=r"(outptr2),
						"=r"(outptr3),
						"=r"(nn),
						"=r"(nn4),
						"=r"(nn1),
						"=r"(tmpptr),
						"=r"(kptr0)
						: "0"(outptr0),
						"1"(outptr1),
						"2"(outptr2),
						"3"(outptr3),
						"4"(nn),
						"5"(nn4),
						"6"(nn1),
						"7"(tmpptr),
						"8"(kptr0)
						: "memory", "x4", "x5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
				}
				for (; i + 7 < size; i += 8)
				{
					const signed char* tmpptr = tmp_data + (i / 16 + (i % 16) / 8) * tmp_cstep;
					const signed char* kptr0 = kernel_tm_gemm_int8_data + (p / 4) * kernel_tm_gemm_int8_cstep;

					int nn = (inch / 8) * maxk;
					int nn4 = ((inch % 8) / 4) * maxk;
					int nn1 = (inch % 4) * maxk;

					int32x4_t _sum0 = vdupq_n_s32(0);
					int32x4_t _sum1 = vdupq_n_s32(0);
					int32x4_t _sum2 = vdupq_n_s32(0);
					int32x4_t _sum3 = vdupq_n_s32(0);
					int32x4_t _sum4 = vdupq_n_s32(0);
					int32x4_t _sum5 = vdupq_n_s32(0);
					int32x4_t _sum6 = vdupq_n_s32(0);
					int32x4_t _sum7 = vdupq_n_s32(0);

					for (int j = 0; j < nn; j++)
					{
						int8x16_t _val0123_l = vld1q_s8(tmpptr);
						int8x16_t _val4567_l = vld1q_s8(tmpptr + 16);

						int8x16_t _w0123_l = vld1q_s8(kptr0);

						_sum0 = vdotq_laneq_s32(_sum0, _w0123_l, _val0123_l, 0);
						_sum1 = vdotq_laneq_s32(_sum1, _w0123_l, _val0123_l, 1);
						_sum2 = vdotq_laneq_s32(_sum2, _w0123_l, _val0123_l, 2);
						_sum3 = vdotq_laneq_s32(_sum3, _w0123_l, _val0123_l, 3);
						_sum4 = vdotq_laneq_s32(_sum4, _w0123_l, _val4567_l, 0);
						_sum5 = vdotq_laneq_s32(_sum5, _w0123_l, _val4567_l, 1);
						_sum6 = vdotq_laneq_s32(_sum6, _w0123_l, _val4567_l, 2);
						_sum7 = vdotq_laneq_s32(_sum7, _w0123_l, _val4567_l, 3);

						int8x16_t _val0123_h = vld1q_s8(tmpptr + 32);
						int8x16_t _val4567_h = vld1q_s8(tmpptr + 48);

						int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);

						_sum0 = vdotq_laneq_s32(_sum0, _w0123_h, _val0123_h, 0);
						_sum1 = vdotq_laneq_s32(_sum1, _w0123_h, _val0123_h, 1);
						_sum2 = vdotq_laneq_s32(_sum2, _w0123_h, _val0123_h, 2);
						_sum3 = vdotq_laneq_s32(_sum3, _w0123_h, _val0123_h, 3);
						_sum4 = vdotq_laneq_s32(_sum4, _w0123_h, _val4567_h, 0);
						_sum5 = vdotq_laneq_s32(_sum5, _w0123_h, _val4567_h, 1);
						_sum6 = vdotq_laneq_s32(_sum6, _w0123_h, _val4567_h, 2);
						_sum7 = vdotq_laneq_s32(_sum7, _w0123_h, _val4567_h, 3);

						tmpptr += 64;
						kptr0 += 32;
					}

					for (int j = 0; j < nn4; j++)
					{
						int8x16_t _val0123 = vld1q_s8(tmpptr);
						int8x16_t _val4567 = vld1q_s8(tmpptr + 16);
						int8x16_t _w0 = vld1q_s8(kptr0);

						_sum0 = vdotq_laneq_s32(_sum0, _w0, _val0123, 0);
						_sum1 = vdotq_laneq_s32(_sum1, _w0, _val0123, 1);
						_sum2 = vdotq_laneq_s32(_sum2, _w0, _val0123, 2);
						_sum3 = vdotq_laneq_s32(_sum3, _w0, _val0123, 3);
						_sum4 = vdotq_laneq_s32(_sum4, _w0, _val4567, 0);
						_sum5 = vdotq_laneq_s32(_sum5, _w0, _val4567, 1);
						_sum6 = vdotq_laneq_s32(_sum6, _w0, _val4567, 2);
						_sum7 = vdotq_laneq_s32(_sum7, _w0, _val4567, 3);

						tmpptr += 32;
						kptr0 += 16;
					}

					int j = 0;
					for (; j + 3 < nn1; j += 4)
					{
						int8x8x4_t _val4 = vld4_s8(tmpptr);

						int8x8x2_t _val0145 = vuzp_s8(_val4.val[0], _val4.val[1]);
						int8x8x2_t _val2367 = vuzp_s8(_val4.val[2], _val4.val[3]);

						int8x16_t _val0123 = vcombine_s8(_val0145.val[0], _val2367.val[0]);
						int8x16_t _val4567 = vcombine_s8(_val0145.val[1], _val2367.val[1]);

						int8x16_t _w = vld1q_s8(kptr0);

						int8x8x2_t _w01 = vuzp_s8(vget_low_s8(_w), vget_high_s8(_w));
						int8x8x2_t _w0123 = vuzp_s8(_w01.val[0], _w01.val[1]);
						int8x16_t _w0123f = vcombine_s8(_w0123.val[0], _w0123.val[1]);

						_sum0 = vdotq_laneq_s32(_sum0, _w0123f, _val0123, 0);
						_sum1 = vdotq_laneq_s32(_sum1, _w0123f, _val0123, 1);
						_sum2 = vdotq_laneq_s32(_sum2, _w0123f, _val0123, 2);
						_sum3 = vdotq_laneq_s32(_sum3, _w0123f, _val0123, 3);
						_sum4 = vdotq_laneq_s32(_sum4, _w0123f, _val4567, 0);
						_sum5 = vdotq_laneq_s32(_sum5, _w0123f, _val4567, 1);
						_sum6 = vdotq_laneq_s32(_sum6, _w0123f, _val4567, 2);
						_sum7 = vdotq_laneq_s32(_sum7, _w0123f, _val4567, 3);

						tmpptr += 32;
						kptr0 += 16;
					}
					for (; j < nn1; j++)
					{
						int16x4_t _val0 = vdup_n_s16(tmpptr[0]);
						int16x4_t _val1 = vdup_n_s16(tmpptr[1]);
						int16x4_t _val2 = vdup_n_s16(tmpptr[2]);
						int16x4_t _val3 = vdup_n_s16(tmpptr[3]);
						int16x4_t _val4 = vdup_n_s16(tmpptr[4]);
						int16x4_t _val5 = vdup_n_s16(tmpptr[5]);
						int16x4_t _val6 = vdup_n_s16(tmpptr[6]);
						int16x4_t _val7 = vdup_n_s16(tmpptr[7]);

						int16x4_t _w0123;
						_w0123 = vset_lane_s16(kptr0[0], _w0123, 0);
						_w0123 = vset_lane_s16(kptr0[1], _w0123, 1);
						_w0123 = vset_lane_s16(kptr0[2], _w0123, 2);
						_w0123 = vset_lane_s16(kptr0[3], _w0123, 3);

						_sum0 = vmlal_s16(_sum0, _val0, _w0123);
						_sum1 = vmlal_s16(_sum1, _val1, _w0123);
						_sum2 = vmlal_s16(_sum2, _val2, _w0123);
						_sum3 = vmlal_s16(_sum3, _val3, _w0123);
						_sum4 = vmlal_s16(_sum4, _val4, _w0123);
						_sum5 = vmlal_s16(_sum5, _val5, _w0123);
						_sum6 = vmlal_s16(_sum6, _val6, _w0123);
						_sum7 = vmlal_s16(_sum7, _val7, _w0123);

						tmpptr += 8;
						kptr0 += 4;
					}

					// transpose 4x8
					int32x4x2_t _s01 = vtrnq_s32(_sum0, _sum1);
					int32x4x2_t _s23 = vtrnq_s32(_sum2, _sum3);
					int32x4x2_t _s45 = vtrnq_s32(_sum4, _sum5);
					int32x4x2_t _s67 = vtrnq_s32(_sum6, _sum7);
					_sum0 = vcombine_s32(vget_low_s32(_s01.val[0]), vget_low_s32(_s23.val[0]));
					_sum1 = vcombine_s32(vget_low_s32(_s01.val[1]), vget_low_s32(_s23.val[1]));
					_sum2 = vcombine_s32(vget_high_s32(_s01.val[0]), vget_high_s32(_s23.val[0]));
					_sum3 = vcombine_s32(vget_high_s32(_s01.val[1]), vget_high_s32(_s23.val[1]));
					_sum4 = vcombine_s32(vget_low_s32(_s45.val[0]), vget_low_s32(_s67.val[0]));
					_sum5 = vcombine_s32(vget_low_s32(_s45.val[1]), vget_low_s32(_s67.val[1]));
					_sum6 = vcombine_s32(vget_high_s32(_s45.val[0]), vget_high_s32(_s67.val[0]));
					_sum7 = vcombine_s32(vget_high_s32(_s45.val[1]), vget_high_s32(_s67.val[1]));

					vst1q_s32(outptr0, _sum0);
					vst1q_s32(outptr1, _sum1);
					vst1q_s32(outptr2, _sum2);
					vst1q_s32(outptr3, _sum3);
					vst1q_s32(outptr0 + 4, _sum4);
					vst1q_s32(outptr1 + 4, _sum5);
					vst1q_s32(outptr2 + 4, _sum6);
					vst1q_s32(outptr3 + 4, _sum7);
					outptr0 += 8;
					outptr1 += 8;
					outptr2 += 8;
					outptr3 += 8;
				}
#endif
				for (; i + 3 < size; i += 4)
				{
#if __ARM_FEATURE_DOTPROD
					const signed char* tmpptr = tmp_data + (i / 16 + (i % 16) / 8 + (i % 8) / 4) * tmp_cstep;
#else
					const signed char* tmpptr = tmp_data + (i / 4) * tmp_cstep;
#endif
					const signed char* kptr0 = kernel_tm_gemm_int8_data + (p / 4) * kernel_tm_gemm_int8_cstep;

					int nn = (inch / 8) * maxk;
					int nn4 = ((inch % 8) / 4) * maxk;
					int nn1 = (inch % 4) * maxk;
#if __ARM_FEATURE_DOTPROD
					int32x4_t _sum0 = vdupq_n_s32(0);
					int32x4_t _sum1 = vdupq_n_s32(0);
					int32x4_t _sum2 = vdupq_n_s32(0);
					int32x4_t _sum3 = vdupq_n_s32(0);

					for (int j = 0; j < nn; j++)
					{
						int8x16_t _val0123_l = vld1q_s8(tmpptr);
						int8x16_t _w0123_l = vld1q_s8(kptr0);

						_sum0 = vdotq_laneq_s32(_sum0, _w0123_l, _val0123_l, 0);
						_sum1 = vdotq_laneq_s32(_sum1, _w0123_l, _val0123_l, 1);
						_sum2 = vdotq_laneq_s32(_sum2, _w0123_l, _val0123_l, 2);
						_sum3 = vdotq_laneq_s32(_sum3, _w0123_l, _val0123_l, 3);

						int8x16_t _val0123_h = vld1q_s8(tmpptr + 16);
						int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);

						_sum0 = vdotq_laneq_s32(_sum0, _w0123_h, _val0123_h, 0);
						_sum1 = vdotq_laneq_s32(_sum1, _w0123_h, _val0123_h, 1);
						_sum2 = vdotq_laneq_s32(_sum2, _w0123_h, _val0123_h, 2);
						_sum3 = vdotq_laneq_s32(_sum3, _w0123_h, _val0123_h, 3);

						tmpptr += 32;
						kptr0 += 32;
					}

					for (int j = 0; j < nn4; j++)
					{
						int8x16_t _val0123 = vld1q_s8(tmpptr);
						int8x16_t _w0 = vld1q_s8(kptr0);

						_sum0 = vdotq_laneq_s32(_sum0, _w0, _val0123, 0);
						_sum1 = vdotq_laneq_s32(_sum1, _w0, _val0123, 1);
						_sum2 = vdotq_laneq_s32(_sum2, _w0, _val0123, 2);
						_sum3 = vdotq_laneq_s32(_sum3, _w0, _val0123, 3);

						tmpptr += 16;
						kptr0 += 16;
					}

					int j = 0;
					for (; j + 3 < nn1; j += 4)
					{
						int8x16_t _val = vld1q_s8(tmpptr);

						int8x8x2_t _val01 = vuzp_s8(vget_low_s8(_val), vget_high_s8(_val));
						int8x8x2_t _val0123 = vuzp_s8(_val01.val[0], _val01.val[1]);
						int8x16_t _val0123f = vcombine_s8(_val0123.val[0], _val0123.val[1]);

						int8x16_t _w = vld1q_s8(kptr0);

						int8x8x2_t _w01 = vuzp_s8(vget_low_s8(_w), vget_high_s8(_w));
						int8x8x2_t _w0123 = vuzp_s8(_w01.val[0], _w01.val[1]);
						int8x16_t _w0123f = vcombine_s8(_w0123.val[0], _w0123.val[1]);

						_sum0 = vdotq_laneq_s32(_sum0, _w0123f, _val0123f, 0);
						_sum1 = vdotq_laneq_s32(_sum1, _w0123f, _val0123f, 1);
						_sum2 = vdotq_laneq_s32(_sum2, _w0123f, _val0123f, 2);
						_sum3 = vdotq_laneq_s32(_sum3, _w0123f, _val0123f, 3);

						tmpptr += 16;
						kptr0 += 16;
					}
					for (; j < nn1; j++)
					{
						int16x4_t _val0 = vdup_n_s16(tmpptr[0]);
						int16x4_t _val1 = vdup_n_s16(tmpptr[1]);
						int16x4_t _val2 = vdup_n_s16(tmpptr[2]);
						int16x4_t _val3 = vdup_n_s16(tmpptr[3]);

						int16x4_t _w0123;
						_w0123 = vset_lane_s16(kptr0[0], _w0123, 0);
						_w0123 = vset_lane_s16(kptr0[1], _w0123, 1);
						_w0123 = vset_lane_s16(kptr0[2], _w0123, 2);
						_w0123 = vset_lane_s16(kptr0[3], _w0123, 3);

						_sum0 = vmlal_s16(_sum0, _val0, _w0123);
						_sum1 = vmlal_s16(_sum1, _val1, _w0123);
						_sum2 = vmlal_s16(_sum2, _val2, _w0123);
						_sum3 = vmlal_s16(_sum3, _val3, _w0123);

						tmpptr += 4;
						kptr0 += 4;
					}

					// transpose 4x4
					int32x4x2_t _s01 = vtrnq_s32(_sum0, _sum1);
					int32x4x2_t _s23 = vtrnq_s32(_sum2, _sum3);
					_sum0 = vcombine_s32(vget_low_s32(_s01.val[0]), vget_low_s32(_s23.val[0]));
					_sum1 = vcombine_s32(vget_low_s32(_s01.val[1]), vget_low_s32(_s23.val[1]));
					_sum2 = vcombine_s32(vget_high_s32(_s01.val[0]), vget_high_s32(_s23.val[0]));
					_sum3 = vcombine_s32(vget_high_s32(_s01.val[1]), vget_high_s32(_s23.val[1]));

					vst1q_s32(outptr0, _sum0);
					vst1q_s32(outptr1, _sum1);
					vst1q_s32(outptr2, _sum2);
					vst1q_s32(outptr3, _sum3);
					outptr0 += 4;
					outptr1 += 4;
					outptr2 += 4;
					outptr3 += 4;
#else  // __ARM_FEATURE_DOTPROD
					asm volatile(
						"eor    v0.16b, v0.16b, v0.16b      \n"
						"eor    v1.16b, v1.16b, v1.16b      \n"
						"eor    v2.16b, v2.16b, v2.16b      \n"
						"eor    v3.16b, v3.16b, v3.16b      \n"

						"cmp    %w4, #0                     \n"
						"beq    3f                          \n"

						"eor    v4.16b, v4.16b, v4.16b      \n"
						"eor    v5.16b, v5.16b, v5.16b      \n"
						"eor    v6.16b, v6.16b, v6.16b      \n"
						"eor    v7.16b, v7.16b, v7.16b      \n"
						"eor    v8.16b, v8.16b, v8.16b      \n"
						"eor    v9.16b, v9.16b, v9.16b      \n"
						"eor    v10.16b, v10.16b, v10.16b   \n"
						"eor    v11.16b, v11.16b, v11.16b   \n"
						"eor    v12.16b, v12.16b, v12.16b   \n"
						"eor    v13.16b, v13.16b, v13.16b   \n"
						"eor    v14.16b, v14.16b, v14.16b   \n"
						"eor    v15.16b, v15.16b, v15.16b   \n"

						"prfm   pldl1keep, [%7, #128]       \n"

						"prfm   pldl1keep, [%8, #256]       \n"

						"lsr    w4, %w4, #1                 \n" // w4 = nn >> 1
						"cmp    w4, #0                      \n"
						"beq    1f                          \n"

						"prfm   pldl1keep, [%8, #512]       \n"

						"add    x5, %7, #16                 \n"

						"prfm   pldl1keep, [x5, #128]       \n"

						"ld1    {v16.16b}, [%7]             \n" // val L H
						"ld1    {v20.16b, v21.16b, v22.16b, v23.16b}, [%8], #64 \n"
						"add    %7, %7, #32                 \n"
						"ext    v17.16b, v16.16b, v16.16b, #8 \n" // val H L

						"ld1    {v18.16b}, [%7]             \n"
						"add    %7, %7, #32                 \n"

						"0:                                 \n"

						"smull  v24.8h, v16.8b,  v20.8b     \n"
						"prfm   pldl1keep, [%8, #256]       \n"
						"smull2 v25.8h, v17.16b, v20.16b    \n"
						"prfm   pldl1keep, [%8, #512]       \n"
						"smull  v26.8h, v16.8b,  v21.8b     \n"
						"subs   w4, w4, #1                  \n"
						"smull2 v27.8h, v17.16b, v21.16b    \n"
						"ext    v19.16b, v18.16b, v18.16b, #8 \n" // val H L

						"smlal  v24.8h, v18.8b,  v22.8b     \n"
						"smlal2 v25.8h, v19.16b, v22.16b    \n"
						"smlal  v26.8h, v18.8b,  v23.8b     \n"
						"smlal2 v27.8h, v19.16b, v23.16b    \n"

						"smull2 v29.8h, v16.16b, v20.16b    \n"
						"sadalp v0.4s, v24.8h               \n"
						"smull  v28.8h, v17.8b,  v20.8b     \n"
						"sadalp v1.4s, v25.8h               \n"
						"smull2 v31.8h, v16.16b, v21.16b    \n"
						"ld1    {v16.16b}, [x5]             \n" // val L H
						"smull  v30.8h, v17.8b,  v21.8b     \n"
						"add    x5, x5, #32                 \n"
						"smlal2 v29.8h, v18.16b, v22.16b    \n"
						"sadalp v2.4s, v26.8h               \n"
						"smlal  v28.8h, v19.8b,  v22.8b     \n"
						"sadalp v3.4s, v27.8h               \n"
						"smlal2 v31.8h, v18.16b, v23.16b    \n"
						"ld1    {v18.16b}, [x5]             \n"
						"smlal  v30.8h, v19.8b,  v23.8b     \n"
						"ext    v17.16b, v16.16b, v16.16b, #8 \n" // val H L

						"smull  v24.8h, v16.8b,  v20.8b     \n"
						"add    x5, x5, #32                 \n"
						"smull2 v25.8h, v17.16b, v20.16b    \n"
						"prfm   pldl1keep, [x5, #128]       \n"
						"smull  v26.8h, v16.8b,  v21.8b     \n"
						"prfm   pldl1keep, [x5, #384]       \n"
						"smull2 v27.8h, v17.16b, v21.16b    \n"
						"ext    v19.16b, v18.16b, v18.16b, #8 \n" // val H L

						"smlal  v24.8h, v18.8b,  v22.8b     \n"
						"sadalp v5.4s, v29.8h               \n"
						"smlal2 v25.8h, v19.16b, v22.16b    \n"
						"sadalp v4.4s, v28.8h               \n"
						"smlal  v26.8h, v18.8b,  v23.8b     \n"
						"sadalp v7.4s, v31.8h               \n"
						"smlal2 v27.8h, v19.16b, v23.16b    \n"
						"sadalp v6.4s, v30.8h               \n"

						"smull2 v29.8h, v16.16b, v20.16b    \n"
						"sadalp v8.4s, v24.8h               \n"
						"smull  v28.8h, v17.8b,  v20.8b     \n"
						"sadalp v9.4s, v25.8h               \n"
						"smull2 v31.8h, v16.16b, v21.16b    \n"
						"ld1    {v16.16b}, [%7]             \n" // val L H
						"smull  v30.8h, v17.8b,  v21.8b     \n"
						"add    %7, %7, #32                 \n"
						"smlal2 v29.8h, v18.16b, v22.16b    \n"
						"sadalp v10.4s, v26.8h              \n"
						"smlal  v28.8h, v19.8b,  v22.8b     \n"
						"sadalp v11.4s, v27.8h              \n"
						"smlal2 v31.8h, v18.16b, v23.16b    \n"
						"ld1    {v18.16b}, [%7]             \n"
						"smlal  v30.8h, v19.8b,  v23.8b     \n"
						"add    %7, %7, #32                 \n"
						"ld1    {v20.16b, v21.16b, v22.16b, v23.16b}, [%8], #64 \n"

						"sadalp v13.4s, v29.8h              \n"
						"prfm   pldl1keep, [%7, #128]       \n"
						"sadalp v12.4s, v28.8h              \n"
						"prfm   pldl1keep, [%7, #384]       \n"
						"sadalp v15.4s, v31.8h              \n"
						"ext    v17.16b, v16.16b, v16.16b, #8 \n" // val H L

						"sadalp v14.4s, v30.8h              \n"

						"bne    0b                          \n"

						"sub    %7, %7, #64                 \n"
						"sub    %8, %8, #64                 \n"

						"1:                                 \n"
						"and    w4, %w4, #1                 \n" // w4 = remain = nn & 1
						"cmp    w4, #0                      \n" // w4 > 0
						"beq    2f                          \n"

						"ld1    {v16.8b, v17.8b}, [%7], #16 \n"
						"ld1    {v20.8b, v21.8b, v22.8b, v23.8b}, [%8], #32 \n"

						"smull  v24.8h, v16.8b, v20.8b      \n"
						"smull  v25.8h, v16.8b, v21.8b      \n"
						"smull  v26.8h, v16.8b, v22.8b      \n"
						"ld1    {v18.8b, v19.8b}, [%7], #16 \n"
						"smull  v27.8h, v16.8b, v23.8b      \n"
						"sadalp v0.4s, v24.8h               \n"
						"smull  v28.8h, v17.8b, v20.8b      \n"
						"sadalp v1.4s, v25.8h               \n"
						"smull  v29.8h, v17.8b, v21.8b      \n"
						"sadalp v2.4s, v26.8h               \n"
						"smull  v30.8h, v17.8b, v22.8b      \n"
						"sadalp v3.4s, v27.8h               \n"
						"smull  v31.8h, v17.8b, v23.8b      \n"
						"sadalp v4.4s, v28.8h               \n"
						"smull  v24.8h, v18.8b, v20.8b      \n"
						"sadalp v5.4s, v29.8h               \n"
						"smull  v25.8h, v18.8b, v21.8b      \n"
						"sadalp v6.4s, v30.8h               \n"
						"smull  v26.8h, v18.8b, v22.8b      \n"
						"sadalp v7.4s, v31.8h               \n"
						"smull  v27.8h, v18.8b, v23.8b      \n"
						"sadalp v8.4s, v24.8h               \n"
						"smull  v28.8h, v19.8b, v20.8b      \n"
						"sadalp v9.4s, v25.8h               \n"
						"smull  v29.8h, v19.8b, v21.8b      \n"
						"sadalp v10.4s, v26.8h              \n"
						"smull  v30.8h, v19.8b, v22.8b      \n"
						"sadalp v11.4s, v27.8h              \n"
						"smull  v31.8h, v19.8b, v23.8b      \n"

						"sadalp v12.4s, v28.8h              \n"
						"sadalp v13.4s, v29.8h              \n"
						"sadalp v14.4s, v30.8h              \n"
						"sadalp v15.4s, v31.8h              \n"

						"2:                                 \n"

						"addp   v0.4s, v0.4s, v1.4s         \n"
						"addp   v2.4s, v2.4s, v3.4s         \n"
						"addp   v4.4s, v4.4s, v5.4s         \n"
						"addp   v6.4s, v6.4s, v7.4s         \n"
						"addp   v8.4s, v8.4s, v9.4s         \n"
						"addp   v10.4s, v10.4s, v11.4s      \n"
						"addp   v12.4s, v12.4s, v13.4s      \n"
						"addp   v14.4s, v14.4s, v15.4s      \n"

						"addp   v0.4s, v0.4s, v2.4s         \n"
						"addp   v1.4s, v4.4s, v6.4s         \n"
						"addp   v2.4s, v8.4s, v10.4s        \n"
						"addp   v3.4s, v12.4s, v14.4s       \n"

						"3:                                 \n"

						"cmp    %w5, #0                     \n"
						"beq    7f                          \n"

						"eor    v8.16b, v8.16b, v8.16b      \n"
						"eor    v9.16b, v9.16b, v9.16b      \n"
						"eor    v10.16b, v10.16b, v10.16b   \n"
						"eor    v11.16b, v11.16b, v11.16b   \n"
						"eor    v12.16b, v12.16b, v12.16b   \n"
						"eor    v13.16b, v13.16b, v13.16b   \n"
						"eor    v14.16b, v14.16b, v14.16b   \n"
						"eor    v15.16b, v15.16b, v15.16b   \n"

						"lsr    w4, %w5, #1                 \n" // w4 = nn4 >> 1
						"cmp    w4, #0                      \n"
						"beq    5f                          \n"

						"4:                                 \n"

						"ld1    {v16.8b, v17.8b}, [%7], #16 \n"
						"ld1    {v22.8b, v23.8b}, [%8], #16 \n"

						"zip1   v18.2s, v16.2s, v16.2s      \n" // _val00
						"zip2   v19.2s, v16.2s, v16.2s      \n" // _val11

						"smull  v24.8h, v18.8b, v22.8b      \n"
						"smull  v25.8h, v18.8b, v23.8b      \n"

						"zip1   v20.2s, v17.2s, v17.2s      \n" // _val22

						"smull  v26.8h, v19.8b, v22.8b      \n"
						"smull  v27.8h, v19.8b, v23.8b      \n"

						"zip2   v21.2s, v17.2s, v17.2s      \n" // _val33

						"smull  v28.8h, v20.8b, v22.8b      \n"
						"smull  v29.8h, v20.8b, v23.8b      \n"

						"ld1    {v16.8b, v17.8b}, [%7], #16 \n"

						"smull  v30.8h, v21.8b, v22.8b      \n"
						"smull  v31.8h, v21.8b, v23.8b      \n"

						"ld1    {v22.8b, v23.8b}, [%8], #16 \n"

						"zip1   v18.2s, v16.2s, v16.2s      \n" // _val44
						"zip2   v19.2s, v16.2s, v16.2s      \n" // _val55

						"smlal  v24.8h, v18.8b, v22.8b      \n"
						"smlal  v25.8h, v18.8b, v23.8b      \n"

						"zip1   v20.2s, v17.2s, v17.2s      \n" // _val66

						"smlal  v26.8h, v19.8b, v22.8b      \n"
						"smlal  v27.8h, v19.8b, v23.8b      \n"

						"zip2   v21.2s, v17.2s, v17.2s      \n" // _val77

						"sadalp v8.4s, v24.8h               \n"
						"smlal  v28.8h, v20.8b, v22.8b      \n"
						"sadalp v9.4s, v25.8h               \n"
						"smlal  v29.8h, v20.8b, v23.8b      \n"
						"sadalp v10.4s, v26.8h              \n"
						"smlal  v30.8h, v21.8b, v22.8b      \n"
						"sadalp v11.4s, v27.8h              \n"
						"smlal  v31.8h, v21.8b, v23.8b      \n"
						"sadalp v12.4s, v28.8h              \n"
						"sadalp v13.4s, v29.8h              \n"

						"subs   w4, w4, #1                  \n"

						"sadalp v14.4s, v30.8h              \n"
						"sadalp v15.4s, v31.8h              \n"

						"bne    4b                          \n"

						"5:                                 \n"

						"and    w4, %w5, #1                 \n" // w4 = remain = nn4 & 1
						"cmp    w4, #0                      \n" // w4 > 0
						"beq    6f                          \n"

						"ld1    {v16.8b, v17.8b}, [%7], #16 \n"
						"ld1    {v22.8b, v23.8b}, [%8], #16 \n"

						"zip1   v18.2s, v16.2s, v16.2s      \n" // _val00
						"zip2   v19.2s, v16.2s, v16.2s      \n" // _val11

						"smull  v24.8h, v18.8b, v22.8b      \n"
						"smull  v25.8h, v18.8b, v23.8b      \n"

						"zip1   v20.2s, v17.2s, v17.2s      \n" // _val22

						"smull  v26.8h, v19.8b, v22.8b      \n"
						"smull  v27.8h, v19.8b, v23.8b      \n"

						"zip2   v21.2s, v17.2s, v17.2s      \n" // _val33

						"sadalp v8.4s, v24.8h               \n"
						"smull  v28.8h, v20.8b, v22.8b      \n"
						"sadalp v9.4s, v25.8h               \n"
						"smull  v29.8h, v20.8b, v23.8b      \n"
						"sadalp v10.4s, v26.8h              \n"
						"smull  v30.8h, v21.8b, v22.8b      \n"
						"sadalp v11.4s, v27.8h              \n"
						"smull  v31.8h, v21.8b, v23.8b      \n"
						"sadalp v12.4s, v28.8h              \n"
						"sadalp v13.4s, v29.8h              \n"
						"sadalp v14.4s, v30.8h              \n"
						"sadalp v15.4s, v31.8h              \n"

						"6:                                 \n"

						"addp   v8.4s, v8.4s, v9.4s         \n"
						"addp   v10.4s, v10.4s, v11.4s      \n"
						"addp   v12.4s, v12.4s, v13.4s      \n"
						"addp   v14.4s, v14.4s, v15.4s      \n"

						"add    v0.4s, v0.4s, v8.4s         \n"
						"add    v1.4s, v1.4s, v10.4s        \n"
						"add    v2.4s, v2.4s, v12.4s        \n"
						"add    v3.4s, v3.4s, v14.4s        \n"

						"7:                                 \n"

						"lsr    w4, %w6, #2                 \n" // w4 = nn1 >> 2
						"cmp    w4, #0                      \n"
						"beq    9f                          \n"

						"8:                                 \n"

						"ld1    {v8.16b}, [%7], #16         \n"
						"ld1    {v9.16b}, [%8], #16         \n"

						"sshll  v4.8h, v8.8b, #0            \n"
						"sshll2 v5.8h, v8.16b, #0           \n"
						"sshll  v6.8h, v9.8b, #0            \n"
						"sshll2 v7.8h, v9.16b, #0           \n"

						"smlal  v0.4s, v6.4h, v4.h[0]       \n"
						"smlal  v1.4s, v6.4h, v4.h[1]       \n"
						"smlal  v2.4s, v6.4h, v4.h[2]       \n"
						"smlal  v3.4s, v6.4h, v4.h[3]       \n"
						"smlal2 v0.4s, v6.8h, v4.h[4]       \n"
						"smlal2 v1.4s, v6.8h, v4.h[5]       \n"
						"smlal2 v2.4s, v6.8h, v4.h[6]       \n"
						"smlal2 v3.4s, v6.8h, v4.h[7]       \n"
						"smlal  v0.4s, v7.4h, v5.h[0]       \n"
						"smlal  v1.4s, v7.4h, v5.h[1]       \n"
						"smlal  v2.4s, v7.4h, v5.h[2]       \n"
						"smlal  v3.4s, v7.4h, v5.h[3]       \n"
						"smlal2 v0.4s, v7.8h, v5.h[4]       \n"
						"smlal2 v1.4s, v7.8h, v5.h[5]       \n"
						"smlal2 v2.4s, v7.8h, v5.h[6]       \n"
						"smlal2 v3.4s, v7.8h, v5.h[7]       \n"

						"subs   w4, w4, #1                  \n"
						"bne    8b                          \n"

						"9:                                 \n"

						"and    w4, %w6, #3                 \n" // w4 = nn1 & 3
						"cmp    w4, #0                      \n" // w4 > 0
						"beq    11f                         \n"

						"10:                                \n"

						"ld1    {v4.8b}, [%7]               \n"
						"ld1    {v6.8b}, [%8]               \n"

						"sshll  v4.8h, v4.8b, #0            \n"
						"sshll  v6.8h, v6.8b, #0            \n"

						"smlal  v0.4s, v6.4h, v4.h[0]       \n"
						"smlal  v1.4s, v6.4h, v4.h[1]       \n"
						"smlal  v2.4s, v6.4h, v4.h[2]       \n"
						"smlal  v3.4s, v6.4h, v4.h[3]       \n"

						"add    %7, %7, #4                  \n"
						"add    %8, %8, #4                  \n"

						"subs   w4, w4, #1                  \n"
						"bne    10b                         \n"

						"11:                                 \n"

						// transpose 4x4
						"trn1   v4.4s, v0.4s, v1.4s         \n"
						"trn2   v5.4s, v0.4s, v1.4s         \n"
						"trn1   v6.4s, v2.4s, v3.4s         \n"
						"trn2   v7.4s, v2.4s, v3.4s         \n"

						"trn1   v0.2d, v4.2d, v6.2d         \n"
						"trn2   v2.2d, v4.2d, v6.2d         \n"
						"trn1   v1.2d, v5.2d, v7.2d         \n"
						"trn2   v3.2d, v5.2d, v7.2d         \n"

						"st1    {v0.4s}, [%0], #16          \n"
						"st1    {v1.4s}, [%1], #16          \n"
						"st1    {v2.4s}, [%2], #16          \n"
						"st1    {v3.4s}, [%3], #16          \n"

						: "=r"(outptr0),
						"=r"(outptr1),
						"=r"(outptr2),
						"=r"(outptr3),
						"=r"(nn),
						"=r"(nn4),
						"=r"(nn1),
						"=r"(tmpptr),
						"=r"(kptr0)
						: "0"(outptr0),
						"1"(outptr1),
						"2"(outptr2),
						"3"(outptr3),
						"4"(nn),
						"5"(nn4),
						"6"(nn1),
						"7"(tmpptr),
						"8"(kptr0)
						: "memory", "x4", "x5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#endif // __ARM_FEATURE_DOTPROD
				}
#endif // __aarch64__
				for (; i + 1 < size; i += 2)
				{
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
					const signed char* tmpptr = tmp_data + (i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2) * tmp_cstep;
#else
					const signed char* tmpptr = tmp_data + (i / 4 + (i % 4) / 2) * tmp_cstep;
#endif
#else
					const signed char* tmpptr = tmp_data + (i / 2) * tmp_cstep;
#endif
					const signed char* kptr0 = kernel_tm_gemm_int8_data + (p / 4) * kernel_tm_gemm_int8_cstep;

					int nn = (inch / 8) * maxk;
					int nn4 = ((inch % 8) / 4) * maxk;
					int nn1 = (inch % 4) * maxk;
#if __aarch64__
					int32x4_t _sum00 = vdupq_n_s32(0);
					int32x4_t _sum10 = vdupq_n_s32(0);
#if __ARM_FEATURE_DOTPROD
					for (int j = 0; j < nn; j++)
					{
						int8x16_t _val01_l_h = vld1q_s8(tmpptr);
						int8x16_t _w0123_l = vld1q_s8(kptr0);

						_sum00 = vdotq_laneq_s32(_sum00, _w0123_l, _val01_l_h, 0);
						_sum10 = vdotq_laneq_s32(_sum10, _w0123_l, _val01_l_h, 1);

						int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);

						_sum00 = vdotq_laneq_s32(_sum00, _w0123_h, _val01_l_h, 2);
						_sum10 = vdotq_laneq_s32(_sum10, _w0123_h, _val01_l_h, 3);

						tmpptr += 16;
						kptr0 += 32;
					}

					if (nn4 > 0)
					{
						int j = 0;
						for (; j + 1 < nn4; j += 2)
						{
							int8x16_t _val0123 = vld1q_s8(tmpptr);
							int8x16_t _w0 = vld1q_s8(kptr0);

							_sum00 = vdotq_laneq_s32(_sum00, _w0, _val0123, 0);
							_sum10 = vdotq_laneq_s32(_sum10, _w0, _val0123, 1);

							int8x16_t _w1 = vld1q_s8(kptr0 + 16);

							_sum00 = vdotq_laneq_s32(_sum00, _w1, _val0123, 2);
							_sum10 = vdotq_laneq_s32(_sum10, _w1, _val0123, 3);

							tmpptr += 16;
							kptr0 += 32;
						}
						for (; j < nn4; j++)
						{
							int8x8_t _val01 = vld1_s8(tmpptr);
							int8x16_t _w0 = vld1q_s8(kptr0);

							_sum00 = vdotq_lane_s32(_sum00, _w0, _val01, 0);
							_sum10 = vdotq_lane_s32(_sum10, _w0, _val01, 1);

							tmpptr += 8;
							kptr0 += 16;
						}
					}
#else  // __ARM_FEATURE_DOTPROD
					if (nn > 0)
					{
						int32x4_t _sum01 = vdupq_n_s32(0);
						int32x4_t _sum02 = vdupq_n_s32(0);
						int32x4_t _sum03 = vdupq_n_s32(0);
						int32x4_t _sum11 = vdupq_n_s32(0);
						int32x4_t _sum12 = vdupq_n_s32(0);
						int32x4_t _sum13 = vdupq_n_s32(0);

						int j = 0;
						for (; j + 1 < nn; j += 2)
						{
							int8x16_t _val0 = vld1q_s8(tmpptr);
							int8x16_t _val1 = vld1q_s8(tmpptr + 16);

							int8x16_t _w01 = vld1q_s8(kptr0);
							int8x16_t _w23 = vld1q_s8(kptr0 + 16);

							int16x8_t _wv00 = vmull_s8(vget_low_s8(_val0), vget_low_s8(_w01));
							int16x8_t _wv01 = vmull_s8(vget_low_s8(_val0), vget_high_s8(_w01));
							int16x8_t _wv02 = vmull_s8(vget_low_s8(_val0), vget_low_s8(_w23));
							int16x8_t _wv03 = vmull_s8(vget_low_s8(_val0), vget_high_s8(_w23));

							int16x8_t _wv10 = vmull_s8(vget_high_s8(_val0), vget_low_s8(_w01));
							int16x8_t _wv11 = vmull_s8(vget_high_s8(_val0), vget_high_s8(_w01));
							int16x8_t _wv12 = vmull_s8(vget_high_s8(_val0), vget_low_s8(_w23));
							int16x8_t _wv13 = vmull_s8(vget_high_s8(_val0), vget_high_s8(_w23));

							int8x16_t _w45 = vld1q_s8(kptr0 + 32);
							int8x16_t _w67 = vld1q_s8(kptr0 + 48);

							_wv00 = vmlal_s8(_wv00, vget_low_s8(_val1), vget_low_s8(_w45));
							_wv01 = vmlal_s8(_wv01, vget_low_s8(_val1), vget_high_s8(_w45));
							_wv02 = vmlal_s8(_wv02, vget_low_s8(_val1), vget_low_s8(_w67));
							_wv03 = vmlal_s8(_wv03, vget_low_s8(_val1), vget_high_s8(_w67));

							_wv10 = vmlal_s8(_wv10, vget_high_s8(_val1), vget_low_s8(_w45));
							_wv11 = vmlal_s8(_wv11, vget_high_s8(_val1), vget_high_s8(_w45));
							_wv12 = vmlal_s8(_wv12, vget_high_s8(_val1), vget_low_s8(_w67));
							_wv13 = vmlal_s8(_wv13, vget_high_s8(_val1), vget_high_s8(_w67));

							_sum00 = vpadalq_s16(_sum00, _wv00);
							_sum01 = vpadalq_s16(_sum01, _wv01);
							_sum02 = vpadalq_s16(_sum02, _wv02);
							_sum03 = vpadalq_s16(_sum03, _wv03);
							_sum10 = vpadalq_s16(_sum10, _wv10);
							_sum11 = vpadalq_s16(_sum11, _wv11);
							_sum12 = vpadalq_s16(_sum12, _wv12);
							_sum13 = vpadalq_s16(_sum13, _wv13);

							tmpptr += 32;
							kptr0 += 64;
						}
						for (; j < nn; j++)
						{
							int8x16_t _val = vld1q_s8(tmpptr);

							int8x16_t _w01 = vld1q_s8(kptr0);
							int8x16_t _w23 = vld1q_s8(kptr0 + 16);

							int16x8_t _wv00 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w01));
							int16x8_t _wv01 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w01));
							int16x8_t _wv02 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w23));
							int16x8_t _wv03 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w23));
							int16x8_t _wv10 = vmull_s8(vget_high_s8(_val), vget_low_s8(_w01));
							int16x8_t _wv11 = vmull_s8(vget_high_s8(_val), vget_high_s8(_w01));
							int16x8_t _wv12 = vmull_s8(vget_high_s8(_val), vget_low_s8(_w23));
							int16x8_t _wv13 = vmull_s8(vget_high_s8(_val), vget_high_s8(_w23));

							_sum00 = vpadalq_s16(_sum00, _wv00);
							_sum01 = vpadalq_s16(_sum01, _wv01);
							_sum02 = vpadalq_s16(_sum02, _wv02);
							_sum03 = vpadalq_s16(_sum03, _wv03);
							_sum10 = vpadalq_s16(_sum10, _wv10);
							_sum11 = vpadalq_s16(_sum11, _wv11);
							_sum12 = vpadalq_s16(_sum12, _wv12);
							_sum13 = vpadalq_s16(_sum13, _wv13);

							tmpptr += 16;
							kptr0 += 32;
						}

						int32x4_t _s001 = vpaddq_s32(_sum00, _sum01);
						int32x4_t _s023 = vpaddq_s32(_sum02, _sum03);
						int32x4_t _s101 = vpaddq_s32(_sum10, _sum11);
						int32x4_t _s123 = vpaddq_s32(_sum12, _sum13);

						_sum00 = vpaddq_s32(_s001, _s023);
						_sum10 = vpaddq_s32(_s101, _s123);
					}

					if (nn4 > 0)
					{
						int32x4_t _sum100 = vdupq_n_s32(0);
						int32x4_t _sum101 = vdupq_n_s32(0);
						int32x4_t _sum110 = vdupq_n_s32(0);
						int32x4_t _sum111 = vdupq_n_s32(0);

						int j = 0;
						for (; j + 1 < nn4; j += 2)
						{
							int8x16_t _val0123 = vld1q_s8(tmpptr);

							int32x4x2_t _val00221133 = vzipq_s32(vreinterpretq_s32_s8(_val0123), vreinterpretq_s32_s8(_val0123));
							int8x8_t _val00 = vreinterpret_s8_s32(vget_low_s32(_val00221133.val[0]));
							int8x8_t _val11 = vreinterpret_s8_s32(vget_high_s32(_val00221133.val[0]));
							int8x8_t _val22 = vreinterpret_s8_s32(vget_low_s32(_val00221133.val[1]));
							int8x8_t _val33 = vreinterpret_s8_s32(vget_high_s32(_val00221133.val[1]));

							int8x16_t _w01 = vld1q_s8(kptr0);
							int8x16_t _w23 = vld1q_s8(kptr0 + 16);

							int16x8_t _wv00 = vmull_s8(_val00, vget_low_s8(_w01));
							int16x8_t _wv01 = vmull_s8(_val00, vget_high_s8(_w01));
							int16x8_t _wv10 = vmull_s8(_val11, vget_low_s8(_w01));
							int16x8_t _wv11 = vmull_s8(_val11, vget_high_s8(_w01));

							_wv00 = vmlal_s8(_wv00, _val22, vget_low_s8(_w23));
							_wv01 = vmlal_s8(_wv01, _val22, vget_high_s8(_w23));
							_wv10 = vmlal_s8(_wv10, _val33, vget_low_s8(_w23));
							_wv11 = vmlal_s8(_wv11, _val33, vget_high_s8(_w23));

							_sum100 = vpadalq_s16(_sum100, _wv00);
							_sum101 = vpadalq_s16(_sum101, _wv01);
							_sum110 = vpadalq_s16(_sum110, _wv10);
							_sum111 = vpadalq_s16(_sum111, _wv11);

							tmpptr += 16;
							kptr0 += 32;
						}
						for (; j < nn4; j++)
						{
							int8x8_t _val01 = vld1_s8(tmpptr);
							int32x2x2_t _val0011 = vzip_s32(vreinterpret_s32_s8(_val01), vreinterpret_s32_s8(_val01));
							int8x8_t _val00 = vreinterpret_s8_s32(_val0011.val[0]);
							int8x8_t _val11 = vreinterpret_s8_s32(_val0011.val[1]);

							int8x16_t _w01 = vld1q_s8(kptr0);

							int16x8_t _wv00 = vmull_s8(_val00, vget_low_s8(_w01));
							int16x8_t _wv01 = vmull_s8(_val00, vget_high_s8(_w01));
							int16x8_t _wv10 = vmull_s8(_val11, vget_low_s8(_w01));
							int16x8_t _wv11 = vmull_s8(_val11, vget_high_s8(_w01));

							_sum100 = vpadalq_s16(_sum100, _wv00);
							_sum101 = vpadalq_s16(_sum101, _wv01);
							_sum110 = vpadalq_s16(_sum110, _wv10);
							_sum111 = vpadalq_s16(_sum111, _wv11);

							tmpptr += 8;
							kptr0 += 16;
						}

						int32x4_t _s001 = vpaddq_s32(_sum100, _sum101);
						int32x4_t _s101 = vpaddq_s32(_sum110, _sum111);

						_sum00 = vaddq_s32(_sum00, _s001);
						_sum10 = vaddq_s32(_sum10, _s101);
					}
#endif // __ARM_FEATURE_DOTPROD

					int j = 0;
					for (; j + 3 < nn1; j += 4)
					{
						int16x8_t _val01234567 = vmovl_s8(vld1_s8(tmpptr));

						int8x16_t _w = vld1q_s8(kptr0);
						int16x8_t _w01234567 = vmovl_s8(vget_low_s8(_w));
						int16x8_t _w89abcdef = vmovl_s8(vget_high_s8(_w));
						int16x4_t _w0123 = vget_low_s16(_w01234567);
						int16x4_t _w4567 = vget_high_s16(_w01234567);
						int16x4_t _w89ab = vget_low_s16(_w89abcdef);
						int16x4_t _wcdef = vget_high_s16(_w89abcdef);

						_sum00 = vmlal_laneq_s16(_sum00, _w0123, _val01234567, 0);
						_sum10 = vmlal_laneq_s16(_sum10, _w0123, _val01234567, 1);
						_sum00 = vmlal_laneq_s16(_sum00, _w4567, _val01234567, 2);
						_sum10 = vmlal_laneq_s16(_sum10, _w4567, _val01234567, 3);
						_sum00 = vmlal_laneq_s16(_sum00, _w89ab, _val01234567, 4);
						_sum10 = vmlal_laneq_s16(_sum10, _w89ab, _val01234567, 5);
						_sum00 = vmlal_laneq_s16(_sum00, _wcdef, _val01234567, 6);
						_sum10 = vmlal_laneq_s16(_sum10, _wcdef, _val01234567, 7);

						tmpptr += 8;
						kptr0 += 16;
					}
					for (; j < nn1; j++)
					{
						int16x4_t _val0 = vdup_n_s16(tmpptr[0]);
						int16x4_t _val1 = vdup_n_s16(tmpptr[1]);

						int16x4_t _w0123;
						_w0123 = vset_lane_s16(kptr0[0], _w0123, 0);
						_w0123 = vset_lane_s16(kptr0[1], _w0123, 1);
						_w0123 = vset_lane_s16(kptr0[2], _w0123, 2);
						_w0123 = vset_lane_s16(kptr0[3], _w0123, 3);

						_sum00 = vmlal_s16(_sum00, _val0, _w0123);
						_sum10 = vmlal_s16(_sum10, _val1, _w0123);

						tmpptr += 2;
						kptr0 += 4;
					}

					vst1q_lane_s32(outptr0, _sum00, 0);
					vst1q_lane_s32(outptr1, _sum00, 1);
					vst1q_lane_s32(outptr2, _sum00, 2);
					vst1q_lane_s32(outptr3, _sum00, 3);
					vst1q_lane_s32(outptr0 + 1, _sum10, 0);
					vst1q_lane_s32(outptr1 + 1, _sum10, 1);
					vst1q_lane_s32(outptr2 + 1, _sum10, 2);
					vst1q_lane_s32(outptr3 + 1, _sum10, 3);
					outptr0 += 2;
					outptr1 += 2;
					outptr2 += 2;
					outptr3 += 2;
#else  // __aarch64__
					asm volatile(
						"veor       q0, q0              \n"
						"veor       q1, q1              \n"
						"veor       q2, q2              \n"
						"veor       q3, q3              \n"
						"veor       q4, q4              \n"
						"veor       q5, q5              \n"
						"veor       q6, q6              \n"
						"veor       q7, q7              \n"

						"cmp        %4, #0              \n"
						"beq        3f                  \n"

						"pld        [%7, #256]          \n"

						"lsr        r4, %4, #1          \n" // r4 = nn = size >> 1
						"cmp        r4, #0              \n"
						"beq        1f                  \n"

						"add        r5, %8, #16         \n"
						"pld        [%8, #128]          \n"
						"mov        r6, #32             \n"
						"pld        [%8, #384]          \n"

						"vld1.s8    {d20-d21}, [%8 :128], r6 \n" // _w01

						"vld1.s8    {d16-d19}, [%7 :128]! \n" // _val0 _val1

						"vld1.s8    {d22-d23}, [%8 :128], r6 \n" // _w45

						"0:                             \n"

						"vmull.s8   q12, d16, d20       \n"
						"pld        [%7, #256]          \n"
						"vmull.s8   q13, d16, d21       \n"
						"pld        [%8, #384]          \n"
						"vmull.s8   q14, d17, d20       \n"
						"vmull.s8   q15, d17, d21       \n"
						"vld1.s8    {d20-d21}, [r5 :128], r6 \n" // _w23

						"vmlal.s8   q12, d18, d22       \n"
						"vmlal.s8   q13, d18, d23       \n"
						"subs       r4, r4, #1          \n"
						"vmlal.s8   q14, d19, d22       \n"
						"vmlal.s8   q15, d19, d23       \n"
						"vld1.s8    {d22-d23}, [r5 :128], r6 \n" // _w67

						"vpadal.s16 q0, q12             \n"
						"vmull.s8   q12, d16, d20       \n"
						"vpadal.s16 q1, q13             \n"
						"vmull.s8   q13, d16, d21       \n"
						"vpadal.s16 q4, q14             \n"
						"vmull.s8   q14, d17, d20       \n"
						"vpadal.s16 q5, q15             \n"
						"vmull.s8   q15, d17, d21       \n"
						"vld1.s8    {d16-d17}, [%7 :128]! \n" // _val0

						"vmlal.s8   q12, d18, d22       \n"
						"vld1.s8    {d20-d21}, [%8 :128], r6 \n" // _w01
						"vmlal.s8   q13, d18, d23       \n"
						"pld        [r5, #128]          \n"
						"vmlal.s8   q14, d19, d22       \n"
						"pld        [r5, #384]          \n"
						"vmlal.s8   q15, d19, d23       \n"
						"vld1.s8    {d18-d19}, [%7 :128]! \n" // _val1

						"vpadal.s16 q2, q12             \n"
						"vld1.s8    {d22-d23}, [%8 :128], r6 \n" // _w45
						"vpadal.s16 q3, q13             \n"
						"pld        [%7, #128]          \n"
						"vpadal.s16 q6, q14             \n"
						"pld        [%8, #128]          \n"
						"vpadal.s16 q7, q15             \n"

						"bne        0b                  \n"

						"sub        %7, %7, #32         \n"
						"sub        %8, %8, #64         \n"

						"1:                             \n"
						"and        r4, %4, #1          \n" // r4 = remain = size & 1
						"cmp        r4, #0              \n" // r4 > 0
						"beq        2f                  \n"

						"vld1.s8    {d16-d17}, [%7 :128]! \n" // _val
						"vld1.s8    {d20-d21}, [%8 :128]! \n" // _w01

						"vmull.s8   q12, d16, d20       \n"

						"vld1.s8    {d22-d23}, [%8 :128]! \n" // _w23
						"vmull.s8   q13, d16, d21       \n"
						"vmull.s8   q14, d17, d20       \n"
						"vmull.s8   q15, d17, d21       \n"

						"vpadal.s16 q0, q12             \n"
						"vmull.s8   q12, d16, d22       \n"
						"vpadal.s16 q1, q13             \n"
						"vmull.s8   q13, d16, d23       \n"
						"vpadal.s16 q4, q14             \n"
						"vmull.s8   q14, d17, d22       \n"
						"vpadal.s16 q5, q15             \n"
						"vmull.s8   q15, d17, d23       \n"

						"vpadal.s16 q2, q12             \n"
						"vpadal.s16 q3, q13             \n"
						"vpadal.s16 q6, q14             \n"
						"vpadal.s16 q7, q15             \n"

						"2:                             \n"

						"vpadd.s32  d16, d0, d1         \n"
						"vpadd.s32  d17, d2, d3         \n"
						"vpadd.s32  d18, d4, d5         \n"
						"vpadd.s32  d19, d6, d7         \n"
						"vpadd.s32  d20, d8, d9         \n"
						"vpadd.s32  d21, d10, d11       \n"
						"vpadd.s32  d22, d12, d13       \n"
						"vpadd.s32  d23, d14, d15       \n"

						"vpadd.s32  d0, d16, d17        \n"
						"vpadd.s32  d1, d18, d19        \n"
						"vpadd.s32  d2, d20, d21        \n"
						"vpadd.s32  d3, d22, d23        \n"

						"3:                             \n"

						"cmp        %5, #0              \n"
						"beq        7f                  \n"

						"veor       q2, q2              \n"
						"veor       q3, q3              \n"
						"veor       q4, q4              \n"
						"veor       q5, q5              \n"

						"lsr        r4, %5, #1          \n" // r4 = nn4 >> 1
						"cmp        r4, #0              \n"
						"beq        5f                  \n"

						"4:                             \n"

						"vld1.s8    {d16-d17}, [%7]!    \n" // _val0123
						"vld1.s8    {d20-d23}, [%8]!    \n" // _w01 _w23

						"vmov.s8    q9, q8              \n"
						"vtrn.s32   q8, q9              \n" // _val00 _val22 _val11 _val33

						"vmull.s8   q12, d16, d20       \n"
						"vmull.s8   q13, d16, d21       \n"
						"vmull.s8   q14, d18, d20       \n"
						"vmull.s8   q15, d18, d21       \n"

						"vmlal.s8   q12, d17, d22       \n"
						"vmlal.s8   q13, d17, d23       \n"
						"vmlal.s8   q14, d19, d22       \n"
						"vmlal.s8   q15, d19, d23       \n"

						"vpadal.s16 q2, q12             \n"
						"vpadal.s16 q3, q13             \n"
						"vpadal.s16 q4, q14             \n"
						"vpadal.s16 q5, q15             \n"

						"subs       r4, r4, #1          \n"
						"bne        4b                  \n"

						"5:                             \n"

						"and        r4, %5, #1          \n" // r4 = nn4 & 1
						"cmp        r4, #0              \n" // r4 > 0
						"beq        6f                  \n"

						"vld1.s8    {d16}, [%7]!        \n" // _val01
						"vld1.s8    {d18-d19}, [%8]!    \n" // _w01

						"vmov.s8    d17, d16            \n"
						"vtrn.s32   d16, d17            \n" // _val00 _val11

						"vmull.s8   q12, d16, d18       \n"
						"vmull.s8   q13, d16, d19       \n"
						"vmull.s8   q14, d17, d18       \n"
						"vmull.s8   q15, d17, d19       \n"

						"vpadal.s16 q2, q12             \n"
						"vpadal.s16 q3, q13             \n"
						"vpadal.s16 q4, q14             \n"
						"vpadal.s16 q5, q15             \n"

						"6:                             \n"

						"vpadd.s32  d16, d4, d5         \n"
						"vpadd.s32  d17, d6, d7         \n"
						"vpadd.s32  d18, d8, d9         \n"
						"vpadd.s32  d19, d10, d11       \n"

						"vadd.s32   q0, q0, q8          \n"
						"vadd.s32   q1, q1, q9          \n"

						"7:                             \n"

						"lsr        r4, %6, #2          \n" // r4 = nn1 >> 2
						"cmp        r4, #0              \n"
						"beq        9f                  \n"

						"8:                             \n"

						"vld1.s8    {d4}, [%7]!         \n"
						"vmovl.s8   q2, d4              \n"

						"vld1.s8    {d10-d11}, [%8]!    \n"
						"vmovl.s8   q3, d10             \n"
						"vmovl.s8   q4, d11             \n"

						"vmlal.s16  q0, d6, d4[0]       \n"
						"vmlal.s16  q1, d6, d4[1]       \n"
						"vmlal.s16  q0, d7, d4[2]       \n"
						"vmlal.s16  q1, d7, d4[3]       \n"
						"vmlal.s16  q0, d8, d5[0]       \n"
						"vmlal.s16  q1, d8, d5[1]       \n"
						"vmlal.s16  q0, d9, d5[2]       \n"
						"vmlal.s16  q1, d9, d5[3]       \n"

						"subs       r4, r4, #1          \n"
						"bne        8b                  \n"

						"9:                             \n"

						"and        r4, %6, #3          \n" // r4 = nn1 & 3
						"cmp        r4, #0              \n" // w4 > 0
						"beq        11f                 \n"

						"10:                            \n"

						"vld1.s8    {d4[]}, [%7]!       \n"
						"vld1.s8    {d6[]}, [%7]!       \n"
						"vmovl.s8   q2, d4              \n"
						"vmovl.s8   q3, d6              \n"

						"vld1.s8    {d8}, [%8]          \n"
						"vmovl.s8   q4, d8              \n"

						"vmlal.s16  q0, d4, d8          \n"
						"vmlal.s16  q1, d6, d8          \n"

						"add        %8, %8, #4          \n"

						"subs       r4, r4, #1          \n"
						"bne        10b                 \n"

						"11:                            \n"

						"vst1.s32   {d0[0]}, [%0]!      \n"
						"vst1.s32   {d0[1]}, [%1]!      \n"
						"vst1.s32   {d1[0]}, [%2]!      \n"
						"vst1.s32   {d1[1]}, [%3]!      \n"
						"vst1.s32   {d2[0]}, [%0]!      \n"
						"vst1.s32   {d2[1]}, [%1]!      \n"
						"vst1.s32   {d3[0]}, [%2]!      \n"
						"vst1.s32   {d3[1]}, [%3]!      \n"

						: "=r"(outptr0),
						"=r"(outptr1),
						"=r"(outptr2),
						"=r"(outptr3),
						"=r"(nn),
						"=r"(nn4),
						"=r"(nn1),
						"=r"(tmpptr),
						"=r"(kptr0)
						: "0"(outptr0),
						"1"(outptr1),
						"2"(outptr2),
						"3"(outptr3),
						"4"(nn),
						"5"(nn4),
						"6"(nn1),
						"7"(tmpptr),
						"8"(kptr0)
						: "memory", "r4", "r5", "r6", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
				}
				for (; i < size; i++)
				{
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
					const signed char* tmpptr = tmp_data + (i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2) * tmp_cstep;
#else
					const signed char* tmpptr = tmp_data + (i / 4 + (i % 4) / 2 + i % 2) * tmp_cstep;
#endif
#else
					const signed char* tmpptr = tmp_data + (i / 2 + i % 2) * tmp_cstep;
#endif
					const signed char* kptr0 = kernel_tm_gemm_int8_data + (p / 4) * kernel_tm_gemm_int8_cstep;

					int nn = (inch / 8) * maxk;
					int nn4 = ((inch % 8) / 4) * maxk;
					int nn1 = (inch % 4) * maxk;

					int32x4_t _sum0 = vdupq_n_s32(0);
#if __ARM_FEATURE_DOTPROD
					for (int j = 0; j < nn; j++)
					{
						int8x8_t _val0_l_h = vld1_s8(tmpptr);

						int8x16_t _w0123_l = vld1q_s8(kptr0);

						_sum0 = vdotq_lane_s32(_sum0, _w0123_l, _val0_l_h, 0);

						int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);

						_sum0 = vdotq_lane_s32(_sum0, _w0123_h, _val0_l_h, 1);

						tmpptr += 8;
						kptr0 += 32;
					}

					if (nn4 > 0)
					{
						int j = 0;
						for (; j + 1 < nn4; j += 2)
						{
							int8x8_t _val01 = vld1_s8(tmpptr);

							int8x16_t _w0 = vld1q_s8(kptr0);

							_sum0 = vdotq_lane_s32(_sum0, _w0, _val01, 0);

							int8x16_t _w1 = vld1q_s8(kptr0 + 16);

							_sum0 = vdotq_lane_s32(_sum0, _w1, _val01, 1);

							tmpptr += 8;
							kptr0 += 32;
						}
						for (; j < nn4; j++)
						{
							int8x8_t _val_xxx = vld1_s8(tmpptr);

							int8x16_t _w0 = vld1q_s8(kptr0);

							_sum0 = vdotq_lane_s32(_sum0, _w0, _val_xxx, 0);

							tmpptr += 4;
							kptr0 += 16;
						}
					}
#else // __ARM_FEATURE_DOTPROD
					if (nn > 0)
					{
						int32x4_t _sum1 = vdupq_n_s32(0);
						int32x4_t _sum2 = vdupq_n_s32(0);
						int32x4_t _sum3 = vdupq_n_s32(0);

						int j = 0;
						for (; j + 1 < nn; j += 2)
						{
							int8x16_t _val = vld1q_s8(tmpptr);

							int8x16_t _w01 = vld1q_s8(kptr0);
							int8x16_t _w23 = vld1q_s8(kptr0 + 16);

							int16x8_t _wv0 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w01));
							int16x8_t _wv1 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w01));
							int16x8_t _wv2 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w23));
							int16x8_t _wv3 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w23));

							int8x16_t _w45 = vld1q_s8(kptr0 + 32);
							int8x16_t _w67 = vld1q_s8(kptr0 + 48);

							_wv0 = vmlal_s8(_wv0, vget_high_s8(_val), vget_low_s8(_w45));
							_wv1 = vmlal_s8(_wv1, vget_high_s8(_val), vget_high_s8(_w45));
							_wv2 = vmlal_s8(_wv2, vget_high_s8(_val), vget_low_s8(_w67));
							_wv3 = vmlal_s8(_wv3, vget_high_s8(_val), vget_high_s8(_w67));

							_sum0 = vpadalq_s16(_sum0, _wv0);
							_sum1 = vpadalq_s16(_sum1, _wv1);
							_sum2 = vpadalq_s16(_sum2, _wv2);
							_sum3 = vpadalq_s16(_sum3, _wv3);

							tmpptr += 16;
							kptr0 += 64;
						}
						for (; j < nn; j++)
						{
							int8x8_t _val = vld1_s8(tmpptr);

							int8x16_t _w01 = vld1q_s8(kptr0);
							int8x16_t _w23 = vld1q_s8(kptr0 + 16);

							int16x8_t _wv0 = vmull_s8(_val, vget_low_s8(_w01));
							int16x8_t _wv1 = vmull_s8(_val, vget_high_s8(_w01));
							int16x8_t _wv2 = vmull_s8(_val, vget_low_s8(_w23));
							int16x8_t _wv3 = vmull_s8(_val, vget_high_s8(_w23));

							_sum0 = vpadalq_s16(_sum0, _wv0);
							_sum1 = vpadalq_s16(_sum1, _wv1);
							_sum2 = vpadalq_s16(_sum2, _wv2);
							_sum3 = vpadalq_s16(_sum3, _wv3);

							tmpptr += 8;
							kptr0 += 32;
						}

#if __aarch64__
						int32x4_t _s01 = vpaddq_s32(_sum0, _sum1);
						int32x4_t _s23 = vpaddq_s32(_sum2, _sum3);

						_sum0 = vpaddq_s32(_s01, _s23);
#else
						int32x2_t _s01_low = vpadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
						int32x2_t _s01_high = vpadd_s32(vget_low_s32(_sum1), vget_high_s32(_sum1));
						int32x2_t _s23_low = vpadd_s32(vget_low_s32(_sum2), vget_high_s32(_sum2));
						int32x2_t _s23_high = vpadd_s32(vget_low_s32(_sum3), vget_high_s32(_sum3));

						_sum0 = vcombine_s32(vpadd_s32(_s01_low, _s01_high), vpadd_s32(_s23_low, _s23_high));
#endif
					}

					if (nn4 > 0)
					{
						int32x4_t _sum10 = vdupq_n_s32(0);
						int32x4_t _sum11 = vdupq_n_s32(0);

						int j = 0;
						for (; j + 1 < nn4; j += 2)
						{
							int8x8_t _val01 = vld1_s8(tmpptr);
							int32x2x2_t _val0011 = vzip_s32(vreinterpret_s32_s8(_val01), vreinterpret_s32_s8(_val01));
							int8x8_t _val00 = vreinterpret_s8_s32(_val0011.val[0]);
							int8x8_t _val11 = vreinterpret_s8_s32(_val0011.val[1]);

							int8x16_t _w0 = vld1q_s8(kptr0);
							int8x16_t _w1 = vld1q_s8(kptr0 + 16);

							int16x8_t _wv0 = vmull_s8(_val00, vget_low_s8(_w0));
							int16x8_t _wv1 = vmull_s8(_val00, vget_high_s8(_w0));

							_wv0 = vmlal_s8(_wv0, _val11, vget_low_s8(_w1));
							_wv1 = vmlal_s8(_wv1, _val11, vget_high_s8(_w1));

							_sum10 = vpadalq_s16(_sum10, _wv0);
							_sum11 = vpadalq_s16(_sum11, _wv1);

							tmpptr += 8;
							kptr0 += 32;
						}
						for (; j < nn4; j++)
						{
							int8x8_t _val_xxx = vld1_s8(tmpptr);
							int8x8_t _val_val = vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_val_xxx), vreinterpret_s32_s8(_val_xxx)).val[0]);

							int8x16_t _w0 = vld1q_s8(kptr0);

							int16x8_t _wv0 = vmull_s8(_val_val, vget_low_s8(_w0));
							int16x8_t _wv1 = vmull_s8(_val_val, vget_high_s8(_w0));

							_sum10 = vpadalq_s16(_sum10, _wv0);
							_sum11 = vpadalq_s16(_sum11, _wv1);

							tmpptr += 4;
							kptr0 += 16;
						}

#if __aarch64__
						int32x4_t _s01 = vpaddq_s32(_sum10, _sum11);
#else
						int32x2_t _s01_low = vpadd_s32(vget_low_s32(_sum10), vget_high_s32(_sum10));
						int32x2_t _s01_high = vpadd_s32(vget_low_s32(_sum11), vget_high_s32(_sum11));

						int32x4_t _s01 = vcombine_s32(_s01_low, _s01_high);
#endif

						_sum0 = vaddq_s32(_sum0, _s01);
					}
#endif // __ARM_FEATURE_DOTPROD

					int32x4_t _sum1 = vdupq_n_s32(0);

					int j = 0;
					for (; j + 3 < nn1; j += 4)
					{
						int16x4_t _val0123 = vget_low_s16(vmovl_s8(vld1_s8(tmpptr)));

						int8x16_t _w = vld1q_s8(kptr0);
						int16x8_t _w01234567 = vmovl_s8(vget_low_s8(_w));
						int16x8_t _w89abcdef = vmovl_s8(vget_high_s8(_w));
						int16x4_t _w0123 = vget_low_s16(_w01234567);
						int16x4_t _w4567 = vget_high_s16(_w01234567);
						int16x4_t _w89ab = vget_low_s16(_w89abcdef);
						int16x4_t _wcdef = vget_high_s16(_w89abcdef);

						_sum0 = vmlal_lane_s16(_sum0, _w0123, _val0123, 0);
						_sum1 = vmlal_lane_s16(_sum1, _w4567, _val0123, 1);
						_sum0 = vmlal_lane_s16(_sum0, _w89ab, _val0123, 2);
						_sum1 = vmlal_lane_s16(_sum1, _wcdef, _val0123, 3);

						tmpptr += 4;
						kptr0 += 16;
					}
					for (; j < nn1; j++)
					{
						int16x4_t _val = vdup_n_s16(tmpptr[0]);

						int16x4_t _w0123;
						_w0123 = vset_lane_s16(kptr0[0], _w0123, 0);
						_w0123 = vset_lane_s16(kptr0[1], _w0123, 1);
						_w0123 = vset_lane_s16(kptr0[2], _w0123, 2);
						_w0123 = vset_lane_s16(kptr0[3], _w0123, 3);

						_sum0 = vmlal_s16(_sum0, _val, _w0123);

						tmpptr += 1;
						kptr0 += 4;
					}

					_sum0 = vaddq_s32(_sum0, _sum1);

					vst1q_lane_s32(outptr0, _sum0, 0);
					vst1q_lane_s32(outptr1, _sum0, 1);
					vst1q_lane_s32(outptr2, _sum0, 2);
					vst1q_lane_s32(outptr3, _sum0, 3);
					outptr0 += 1;
					outptr1 += 1;
					outptr2 += 1;
					outptr3 += 1;
				}
			}

			remain_outch_start += nn_outch << 2;
#endif // __ARM_NEON

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
			for (int p = remain_outch_start; p < outch; p++)
			{
				int* outptr0 = top_data + (p)*outw * outh;

				int i = 0;
#if __ARM_NEON
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
				for (; i + 15 < size; i += 16)
				{
					const signed char* tmpptr = tmp_data + (i / 16) * tmp_cstep;
					const signed char* kptr0 = kernel_tm_gemm_int8_data + (p / 4 + p % 4) * kernel_tm_gemm_int8_cstep;

					int nn = (inch / 8) * maxk;
					int nn4 = ((inch % 8) / 4) * maxk;
					int nn1 = (inch % 4) * maxk;

					int32x4_t _sum0 = vdupq_n_s32(0);
					int32x4_t _sum1 = vdupq_n_s32(0);
					int32x4_t _sum2 = vdupq_n_s32(0);
					int32x4_t _sum3 = vdupq_n_s32(0);

					for (int j = 0; j < nn; j++)
					{
						int8x16_t _val0123_l = vld1q_s8(tmpptr);
						int8x16_t _val4567_l = vld1q_s8(tmpptr + 16);
						int8x16_t _val89ab_l = vld1q_s8(tmpptr + 32);
						int8x16_t _valcdef_l = vld1q_s8(tmpptr + 48);
						int8x16_t _val0123_h = vld1q_s8(tmpptr + 64);
						int8x16_t _val4567_h = vld1q_s8(tmpptr + 80);
						int8x16_t _val89ab_h = vld1q_s8(tmpptr + 96);
						int8x16_t _valcdef_h = vld1q_s8(tmpptr + 112);
						int8x8_t _w_lh = vld1_s8(kptr0);

						_sum0 = vdotq_lane_s32(_sum0, _val0123_l, _w_lh, 0);
						_sum1 = vdotq_lane_s32(_sum1, _val4567_l, _w_lh, 0);
						_sum2 = vdotq_lane_s32(_sum2, _val89ab_l, _w_lh, 0);
						_sum3 = vdotq_lane_s32(_sum3, _valcdef_l, _w_lh, 0);
						_sum0 = vdotq_lane_s32(_sum0, _val0123_h, _w_lh, 1);
						_sum1 = vdotq_lane_s32(_sum1, _val4567_h, _w_lh, 1);
						_sum2 = vdotq_lane_s32(_sum2, _val89ab_h, _w_lh, 1);
						_sum3 = vdotq_lane_s32(_sum3, _valcdef_h, _w_lh, 1);

						tmpptr += 128;
						kptr0 += 8;
					}

					if (nn4 > 0)
					{
						int32x4_t _sum4 = vdupq_n_s32(0);
						int32x4_t _sum5 = vdupq_n_s32(0);
						int32x4_t _sum6 = vdupq_n_s32(0);
						int32x4_t _sum7 = vdupq_n_s32(0);

						for (int j = 0; j < nn4; j++)
						{
							int8x16_t _val0 = vld1q_s8(tmpptr);
							int8x16_t _val1 = vld1q_s8(tmpptr + 16);
							int8x16_t _val2 = vld1q_s8(tmpptr + 32);
							int8x16_t _val3 = vld1q_s8(tmpptr + 48);

							int8x8_t _w_0123_xxxx = vld1_s8(kptr0);

							_sum4 = vdotq_lane_s32(_sum4, _val0, _w_0123_xxxx, 0);
							_sum5 = vdotq_lane_s32(_sum5, _val1, _w_0123_xxxx, 0);
							_sum6 = vdotq_lane_s32(_sum6, _val2, _w_0123_xxxx, 0);
							_sum7 = vdotq_lane_s32(_sum7, _val3, _w_0123_xxxx, 0);

							tmpptr += 64;
							kptr0 += 4;
						}

						_sum0 = vaddq_s32(_sum0, _sum4);
						_sum1 = vaddq_s32(_sum1, _sum5);
						_sum2 = vaddq_s32(_sum2, _sum6);
						_sum3 = vaddq_s32(_sum3, _sum7);
					}

					int j = 0;
					for (; j < nn1; j++)
					{
						int8x16_t _val = vld1q_s8(tmpptr);
						int8x8_t _w = vld1_dup_s8(kptr0);

						int16x8_t _s0 = vmull_s8(vget_low_s8(_val), _w);
						int16x8_t _s1 = vmull_s8(vget_high_s8(_val), _w);

						_sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
						_sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
						_sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
						_sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));

						tmpptr += 16;
						kptr0 += 1;
					}

					vst1q_s32(outptr0, _sum0);
					vst1q_s32(outptr0 + 4, _sum1);
					vst1q_s32(outptr0 + 8, _sum2);
					vst1q_s32(outptr0 + 12, _sum3);
					outptr0 += 16;
				}
				for (; i + 7 < size; i += 8)
				{
					const signed char* tmpptr = tmp_data + (i / 16 + (i % 16) / 8) * tmp_cstep;
					const signed char* kptr0 = kernel_tm_gemm_int8_data + (p / 4 + p % 4) * kernel_tm_gemm_int8_cstep;

					int nn = (inch / 8) * maxk;
					int nn4 = ((inch % 8) / 4) * maxk;
					int nn1 = (inch % 4) * maxk;

					int32x4_t _sum0 = vdupq_n_s32(0);
					int32x4_t _sum1 = vdupq_n_s32(0);
					if (nn > 0)
					{
						int32x4_t _sum2 = vdupq_n_s32(0);
						int32x4_t _sum3 = vdupq_n_s32(0);

						for (int j = 0; j < nn; j++)
						{
							int8x16_t _val0123_l = vld1q_s8(tmpptr);
							int8x16_t _val4567_l = vld1q_s8(tmpptr + 16);
							int8x16_t _val0123_h = vld1q_s8(tmpptr + 32);
							int8x16_t _val4567_h = vld1q_s8(tmpptr + 48);
							int8x8_t _w_lh = vld1_s8(kptr0);

							_sum0 = vdotq_lane_s32(_sum0, _val0123_l, _w_lh, 0);
							_sum1 = vdotq_lane_s32(_sum1, _val4567_l, _w_lh, 0);
							_sum2 = vdotq_lane_s32(_sum2, _val0123_h, _w_lh, 1);
							_sum3 = vdotq_lane_s32(_sum3, _val4567_h, _w_lh, 1);

							tmpptr += 64;
							kptr0 += 8;
						}

						_sum0 = vaddq_s32(_sum0, _sum2);
						_sum1 = vaddq_s32(_sum1, _sum3);
					}

					if (nn4 > 0)
					{
						int32x4_t _sum2 = vdupq_n_s32(0);
						int32x4_t _sum3 = vdupq_n_s32(0);

						for (int j = 0; j < nn4; j++)
						{
							int8x16_t _val0 = vld1q_s8(tmpptr);
							int8x16_t _val1 = vld1q_s8(tmpptr + 16);

							int8x8_t _w_0123_xxxx = vld1_s8(kptr0);

							_sum2 = vdotq_lane_s32(_sum2, _val0, _w_0123_xxxx, 0);
							_sum3 = vdotq_lane_s32(_sum3, _val1, _w_0123_xxxx, 0);

							tmpptr += 32;
							kptr0 += 4;
						}

						_sum0 = vaddq_s32(_sum0, _sum2);
						_sum1 = vaddq_s32(_sum1, _sum3);
					}

					int j = 0;
					for (; j < nn1; j++)
					{
						int8x8_t _val = vld1_s8(tmpptr);
						int8x8_t _w = vld1_dup_s8(kptr0);

						int16x8_t _s = vmull_s8(_val, _w);

						_sum0 = vaddw_s16(_sum0, vget_low_s16(_s));
						_sum1 = vaddw_s16(_sum1, vget_high_s16(_s));

						tmpptr += 8;
						kptr0 += 1;
					}

					vst1q_s32(outptr0, _sum0);
					vst1q_s32(outptr0 + 4, _sum1);
					outptr0 += 8;
				}
#endif // __ARM_FEATURE_DOTPROD
				for (; i + 3 < size; i += 4)
				{
#if __ARM_FEATURE_DOTPROD
					const signed char* tmpptr = tmp_data + (i / 16 + (i % 16) / 8 + (i % 8) / 4) * tmp_cstep;
#else
					const signed char* tmpptr = tmp_data + (i / 4) * tmp_cstep;
#endif
					const signed char* kptr0 = kernel_tm_gemm_int8_data + (p / 4 + p % 4) * kernel_tm_gemm_int8_cstep;

					int nn = (inch / 8) * maxk;
					int nn4 = ((inch % 8) / 4) * maxk;
					int nn1 = (inch % 4) * maxk;

					int32x4_t _sum0 = vdupq_n_s32(0);
					if (nn > 0)
					{
#if __ARM_FEATURE_DOTPROD
						int32x4_t _sum1 = vdupq_n_s32(0);

						int j = 0;
						for (; j < nn; j++)
						{
							int8x16_t _val0123_l = vld1q_s8(tmpptr);
							int8x16_t _val0123_h = vld1q_s8(tmpptr + 16);
							int8x8_t _w_lh = vld1_s8(kptr0);

							_sum0 = vdotq_lane_s32(_sum0, _val0123_l, _w_lh, 0);
							_sum1 = vdotq_lane_s32(_sum1, _val0123_h, _w_lh, 1);

							tmpptr += 32;
							kptr0 += 8;
						}

						_sum0 = vaddq_s32(_sum0, _sum1);
#else  // __ARM_FEATURE_DOTPROD
						int32x4_t _sum1 = vdupq_n_s32(0);
						int32x4_t _sum2 = vdupq_n_s32(0);
						int32x4_t _sum3 = vdupq_n_s32(0);
						int32x4_t _sum4 = vdupq_n_s32(0);
						int32x4_t _sum5 = vdupq_n_s32(0);
						int32x4_t _sum6 = vdupq_n_s32(0);
						int32x4_t _sum7 = vdupq_n_s32(0);

						int j = 0;
						for (; j + 1 < nn; j += 2)
						{
							int8x16_t _val0 = vld1q_s8(tmpptr);
							int8x16_t _val1 = vld1q_s8(tmpptr + 16);
							int8x16_t _val2 = vld1q_s8(tmpptr + 32);
							int8x16_t _val3 = vld1q_s8(tmpptr + 48);
							int8x16_t _w = vld1q_s8(kptr0);

							int16x8_t _s0 = vmull_s8(vget_low_s8(_val0), vget_low_s8(_w));
							int16x8_t _s1 = vmull_s8(vget_high_s8(_val0), vget_low_s8(_w));
							int16x8_t _s2 = vmull_s8(vget_low_s8(_val1), vget_low_s8(_w));
							int16x8_t _s3 = vmull_s8(vget_high_s8(_val1), vget_low_s8(_w));

							_s0 = vmlal_s8(_s0, vget_low_s8(_val2), vget_high_s8(_w));
							_s1 = vmlal_s8(_s1, vget_high_s8(_val2), vget_high_s8(_w));
							_s2 = vmlal_s8(_s2, vget_low_s8(_val3), vget_high_s8(_w));
							_s3 = vmlal_s8(_s3, vget_high_s8(_val3), vget_high_s8(_w));

							_sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
							_sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
							_sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
							_sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));
							_sum4 = vaddw_s16(_sum4, vget_low_s16(_s2));
							_sum5 = vaddw_s16(_sum5, vget_high_s16(_s2));
							_sum6 = vaddw_s16(_sum6, vget_low_s16(_s3));
							_sum7 = vaddw_s16(_sum7, vget_high_s16(_s3));

							tmpptr += 64;
							kptr0 += 16;
						}
						for (; j < nn; j++)
						{
							int8x16_t _val0 = vld1q_s8(tmpptr);
							int8x16_t _val1 = vld1q_s8(tmpptr + 16);
							int8x8_t _w = vld1_s8(kptr0);

							int16x8_t _s0 = vmull_s8(vget_low_s8(_val0), _w);
							int16x8_t _s1 = vmull_s8(vget_high_s8(_val0), _w);
							int16x8_t _s2 = vmull_s8(vget_low_s8(_val1), _w);
							int16x8_t _s3 = vmull_s8(vget_high_s8(_val1), _w);

							_sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
							_sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
							_sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
							_sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));
							_sum4 = vaddw_s16(_sum4, vget_low_s16(_s2));
							_sum5 = vaddw_s16(_sum5, vget_high_s16(_s2));
							_sum6 = vaddw_s16(_sum6, vget_low_s16(_s3));
							_sum7 = vaddw_s16(_sum7, vget_high_s16(_s3));

							tmpptr += 32;
							kptr0 += 8;
						}

						_sum0 = vaddq_s32(_sum0, _sum1);
						_sum2 = vaddq_s32(_sum2, _sum3);
						_sum4 = vaddq_s32(_sum4, _sum5);
						_sum6 = vaddq_s32(_sum6, _sum7);

						int32x2_t _s0 = vadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
						int32x2_t _s2 = vadd_s32(vget_low_s32(_sum2), vget_high_s32(_sum2));
						int32x2_t _s4 = vadd_s32(vget_low_s32(_sum4), vget_high_s32(_sum4));
						int32x2_t _s6 = vadd_s32(vget_low_s32(_sum6), vget_high_s32(_sum6));
						int32x2_t _ss0 = vpadd_s32(_s0, _s2);
						int32x2_t _ss1 = vpadd_s32(_s4, _s6);
						_sum0 = vcombine_s32(_ss0, _ss1);
#endif // __ARM_FEATURE_DOTPROD
					}

					int sum0123[4] = { 0, 0, 0, 0 };

					if (nn4 > 0)
					{
#if __ARM_FEATURE_DOTPROD
						int32x4_t _sum1 = vdupq_n_s32(0);

						int j = 0;
						for (; j < nn4; j++)
						{
							int8x16_t _val0123_lh = vld1q_s8(tmpptr);
							int8x8_t _w_lh_xx = vld1_s8(kptr0);

							_sum1 = vdotq_lane_s32(_sum1, _val0123_lh, _w_lh_xx, 0);

							tmpptr += 16;
							kptr0 += 4;
						}

						_sum0 = vaddq_s32(_sum0, _sum1);
#else  // __ARM_FEATURE_DOTPROD
						int j = 0;
						for (; j < nn4; j++)
						{
							signed char val0 = tmpptr[0];
							signed char val1 = tmpptr[1];
							signed char val2 = tmpptr[2];
							signed char val3 = tmpptr[3];
							signed char val4 = tmpptr[4];
							signed char val5 = tmpptr[5];
							signed char val6 = tmpptr[6];
							signed char val7 = tmpptr[7];
							signed char val8 = tmpptr[8];
							signed char val9 = tmpptr[9];
							signed char val10 = tmpptr[10];
							signed char val11 = tmpptr[11];
							signed char val12 = tmpptr[12];
							signed char val13 = tmpptr[13];
							signed char val14 = tmpptr[14];
							signed char val15 = tmpptr[15];

							signed char w0 = kptr0[0];
							signed char w1 = kptr0[1];
							signed char w2 = kptr0[2];
							signed char w3 = kptr0[3];

							sum0123[0] += val0 * w0;
							sum0123[0] += val1 * w1;
							sum0123[0] += val2 * w2;
							sum0123[0] += val3 * w3;
							sum0123[1] += val4 * w0;
							sum0123[1] += val5 * w1;
							sum0123[1] += val6 * w2;
							sum0123[1] += val7 * w3;
							sum0123[2] += val8 * w0;
							sum0123[2] += val9 * w1;
							sum0123[2] += val10 * w2;
							sum0123[2] += val11 * w3;
							sum0123[3] += val12 * w0;
							sum0123[3] += val13 * w1;
							sum0123[3] += val14 * w2;
							sum0123[3] += val15 * w3;

							tmpptr += 16;
							kptr0 += 4;
						}
#endif // __ARM_FEATURE_DOTPROD
					}

					int j = 0;
					for (; j < nn1; j++)
					{
						signed char val0 = tmpptr[0];
						signed char val1 = tmpptr[1];
						signed char val2 = tmpptr[2];
						signed char val3 = tmpptr[3];
						signed char w = kptr0[0];

						sum0123[0] += val0 * w;
						sum0123[1] += val1 * w;
						sum0123[2] += val2 * w;
						sum0123[3] += val3 * w;

						tmpptr += 4;
						kptr0 += 1;
					}

					_sum0 = vaddq_s32(_sum0, vld1q_s32(sum0123));

					vst1q_s32(outptr0, _sum0);
					outptr0 += 4;
				}
#endif // __aarch64__
				for (; i + 1 < size; i += 2)
				{
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
					const signed char* tmpptr = tmp_data + (i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2) * tmp_cstep;
#else
					const signed char* tmpptr = tmp_data + (i / 4 + (i % 4) / 2) * tmp_cstep;
#endif
#else
					const signed char* tmpptr = tmp_data + (i / 2) * tmp_cstep;
#endif
					const signed char* kptr0 = kernel_tm_gemm_int8_data + (p / 4 + p % 4) * kernel_tm_gemm_int8_cstep;

					int nn = (inch / 8) * maxk;
					int nn4 = ((inch % 8) / 4) * maxk;
					int nn1 = (inch % 4) * maxk;

					int32x2_t _sum = vdup_n_s32(0);
					if (nn > 0)
					{
#if __ARM_FEATURE_DOTPROD
						int32x2_t _sum0 = vdup_n_s32(0);
						int32x2_t _sum1 = vdup_n_s32(0);

						int j = 0;
						for (; j < nn; j++)
						{
							int8x16_t _val01_lh = vld1q_s8(tmpptr);
							int8x8_t _w_lh = vld1_s8(kptr0);

							_sum0 = vdot_lane_s32(_sum0, vget_low_s8(_val01_lh), _w_lh, 0);
							_sum1 = vdot_lane_s32(_sum1, vget_high_s8(_val01_lh), _w_lh, 1);

							tmpptr += 16;
							kptr0 += 8;
						}

						_sum = vadd_s32(_sum0, _sum1);
#else  // __ARM_FEATURE_DOTPROD
						int32x4_t _sum0 = vdupq_n_s32(0);
						int32x4_t _sum1 = vdupq_n_s32(0);
						int32x4_t _sum2 = vdupq_n_s32(0);
						int32x4_t _sum3 = vdupq_n_s32(0);

						int j = 0;
						for (; j + 1 < nn; j += 2)
						{
							int8x16_t _val0 = vld1q_s8(tmpptr);
							int8x16_t _val1 = vld1q_s8(tmpptr + 16);
							int8x16_t _w = vld1q_s8(kptr0);

							int16x8_t _s0 = vmull_s8(vget_low_s8(_val0), vget_low_s8(_w));
							int16x8_t _s1 = vmull_s8(vget_high_s8(_val0), vget_low_s8(_w));

							_s0 = vmlal_s8(_s0, vget_low_s8(_val1), vget_high_s8(_w));
							_s1 = vmlal_s8(_s1, vget_high_s8(_val1), vget_high_s8(_w));

							_sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
							_sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
							_sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
							_sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));

							tmpptr += 32;
							kptr0 += 16;
						}
						for (; j < nn; j++)
						{
							int8x16_t _val = vld1q_s8(tmpptr);
							int8x8_t _w = vld1_s8(kptr0);

							int16x8_t _s0 = vmull_s8(vget_low_s8(_val), _w);
							int16x8_t _s1 = vmull_s8(vget_high_s8(_val), _w);

							_sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
							_sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
							_sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
							_sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));

							tmpptr += 16;
							kptr0 += 8;
						}

						_sum0 = vaddq_s32(_sum0, _sum1);
						_sum2 = vaddq_s32(_sum2, _sum3);

						int32x2_t _s0 = vadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
						int32x2_t _s2 = vadd_s32(vget_low_s32(_sum2), vget_high_s32(_sum2));
						_sum = vpadd_s32(_s0, _s2);
#endif // __ARM_FEATURE_DOTPROD
					}

					int sum01[2] = { 0, 0 };

					if (nn4 > 0)
					{
						int j = 0;
						for (; j < nn4; j++)
						{
							signed char val0 = tmpptr[0];
							signed char val1 = tmpptr[1];
							signed char val2 = tmpptr[2];
							signed char val3 = tmpptr[3];
							signed char val4 = tmpptr[4];
							signed char val5 = tmpptr[5];
							signed char val6 = tmpptr[6];
							signed char val7 = tmpptr[7];

							signed char w0 = kptr0[0];
							signed char w1 = kptr0[1];
							signed char w2 = kptr0[2];
							signed char w3 = kptr0[3];

							sum01[0] += val0 * w0;
							sum01[0] += val1 * w1;
							sum01[0] += val2 * w2;
							sum01[0] += val3 * w3;
							sum01[1] += val4 * w0;
							sum01[1] += val5 * w1;
							sum01[1] += val6 * w2;
							sum01[1] += val7 * w3;

							tmpptr += 8;
							kptr0 += 4;
						}
					}

					int j = 0;
					for (; j < nn1; j++)
					{
						signed char val0 = tmpptr[0];
						signed char val1 = tmpptr[1];
						signed char w = kptr0[0];

						sum01[0] += val0 * w;
						sum01[1] += val1 * w;

						tmpptr += 2;
						kptr0 += 1;
					}

					_sum = vadd_s32(_sum, vld1_s32(sum01));

					vst1_s32(outptr0, _sum);
					outptr0 += 2;
				}
				for (; i < size; i++)
				{
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
					const signed char* tmpptr = tmp_data + (i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2) * tmp_cstep;
#else
					const signed char* tmpptr = tmp_data + (i / 4 + (i % 4) / 2 + i % 2) * tmp_cstep;
#endif
#else
					const signed char* tmpptr = tmp_data + (i / 2 + i % 2) * tmp_cstep;
#endif
					const signed char* kptr0 = kernel_tm_gemm_int8_data + (p / 4 + p % 4) * kernel_tm_gemm_int8_cstep;

					int nn = (inch / 8) * maxk;
					int nn4 = ((inch % 8) / 4) * maxk;
					int nn1 = (inch % 4) * maxk;

					int sum = 0;
					if (nn > 0)
					{
#if __ARM_FEATURE_DOTPROD
						int32x4_t _sum0 = vdupq_n_s32(0);
						int32x2_t _sum1 = vdup_n_s32(0);

						int j = 0;
						for (; j + 1 < nn; j += 2)
						{
							int8x16_t _val = vld1q_s8(tmpptr);
							int8x16_t _w = vld1q_s8(kptr0);

							_sum0 = vdotq_s32(_sum0, _val, _w);

							tmpptr += 16;
							kptr0 += 16;
						}
						for (; j < nn; j++)
						{
							int8x8_t _val = vld1_s8(tmpptr);
							int8x8_t _w = vld1_s8(kptr0);

							_sum1 = vdot_s32(_sum1, _val, _w);

							tmpptr += 8;
							kptr0 += 8;
						}

						sum = vaddvq_s32(_sum0) + vaddv_s32(_sum1);
#else // __ARM_FEATURE_DOTPROD
						int32x4_t _sum0 = vdupq_n_s32(0);
						int32x4_t _sum1 = vdupq_n_s32(0);

						int j = 0;
						for (; j + 1 < nn; j += 2)
						{
							int8x16_t _val = vld1q_s8(tmpptr);
							int8x16_t _w = vld1q_s8(kptr0);

							int16x8_t _s8 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w));
							_s8 = vmlal_s8(_s8, vget_high_s8(_val), vget_high_s8(_w));

							_sum0 = vaddw_s16(_sum0, vget_low_s16(_s8));
							_sum1 = vaddw_s16(_sum1, vget_high_s16(_s8));

							tmpptr += 16;
							kptr0 += 16;
						}
						for (; j < nn; j++)
						{
							int8x8_t _val = vld1_s8(tmpptr);
							int8x8_t _w = vld1_s8(kptr0);

							int16x8_t _s8 = vmull_s8(_val, _w);

							_sum0 = vaddw_s16(_sum0, vget_low_s16(_s8));
							_sum1 = vaddw_s16(_sum1, vget_high_s16(_s8));

							tmpptr += 8;
							kptr0 += 8;
						}

						int32x4_t _sum = vaddq_s32(_sum0, _sum1);
#if __aarch64__
						sum = vaddvq_s32(_sum); // dot
#else
						int32x2_t _ss = vadd_s32(vget_low_s32(_sum), vget_high_s32(_sum));
						_ss = vpadd_s32(_ss, _ss);
						sum = vget_lane_s32(_ss, 0);
#endif
#endif // __ARM_FEATURE_DOTPROD
					}

					if (nn4 > 0)
					{
						int j = 0;
						for (; j < nn4; j++)
						{
							signed char val0 = tmpptr[0];
							signed char val1 = tmpptr[1];
							signed char val2 = tmpptr[2];
							signed char val3 = tmpptr[3];

							signed char w0 = kptr0[0];
							signed char w1 = kptr0[1];
							signed char w2 = kptr0[2];
							signed char w3 = kptr0[3];

							sum += val0 * w0;
							sum += val1 * w1;
							sum += val2 * w2;
							sum += val3 * w3;

							tmpptr += 4;
							kptr0 += 4;
						}
					}

					int j = 0;
					for (; j < nn1; j++)
					{
						signed char val = tmpptr[0];
						signed char w = kptr0[0];

						sum += val * w;

						tmpptr += 1;
						kptr0 += 1;
					}

					outptr0[0] = sum;
					outptr0 += 1;
				}
#else  // __ARM_NEON
				for (; i < size; i++)
				{
					const signed char* tmpptr = tmp_data + (i)*tmp_cstep;
					const signed char* kptr0 = kernel_tm_gemm_int8_data + (p)*kernel_tm_gemm_int8_cstep;

					int nn1 = inch * maxk;

					int sum = 0;
					int j = 0;
					for (; j < nn1; j++)
					{
						signed char val = tmpptr[0];
						signed char w = kptr0[0];

						sum += val * w;

						tmpptr += 1;
						kptr0 += 1;
					}

					outptr0[0] = sum;
					outptr0 += 1;
				}
#endif // __ARM_NEON
			}
		}

		INSTANCE_CLASS(operation_convolution_arm);
		REGISTE(operation_convolution_arm);
	}
}