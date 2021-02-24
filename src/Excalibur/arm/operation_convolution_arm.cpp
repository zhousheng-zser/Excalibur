#include "../../../include/Excalibur/arm/operation_convolution_arm.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"
//#include "../../include/Excalibur/im2col.hpp"
#include "../../include/Excalibur/math_functions.hpp"
#include "../../include/Excalibur/operation_make_border.hpp"
#include "../../include/Excalibur/operation_cut_border.hpp"
//#include "../../include/Primitives/simd_types.hpp"
#include <random>
#include "../../../include/Excalibur/arm/gemm_symm_int8.h"

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
					if ((this->kernel_size_h_ == 1 && this->kernel_size_w_ == 1) && (this->stride_h_ == 1 && this->stride_w_ == 1)/* &&(this->input_channel_ >= 64 && output_channel_ >= 64)*/)
					{
						conv1x1s1_sgemm_transform_kernel_neon();
					}
					else if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 1 && this->stride_w_ == 1) && this->output_channel_<512)
					{
						conv3x3s1_winograd64_transform_kernel_neon5();
					}
					else if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 2 && this->stride_w_ == 2))
					{
						conv3x3s2_transform_kernel_neon();
					}

					conv_im2col_sgemm_transform_kernel_neon();
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
					if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 1 && this->stride_w_ == 1))
					{
						conv3x3s1_winograd43_transform_kernel_int8_neon();
					}
					else if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 2 && this->stride_w_ == 2))
					{
						conv3x3s2_transform_kernel_int8_neon();
					}
					else if ((this->kernel_size_h_ == 1 && this->kernel_size_w_ == 1) && (this->stride_h_ == 1 && this->stride_w_ == 1))
					{
						conv1x1s1_sgemm_transform_kernel_int8_neon();
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
					if ((this->kernel_size_h_ == 1 && this->kernel_size_w_ == 1) && (this->stride_h_ == 1 && this->stride_w_ == 1)/* && (this->input_channel_ >= 64 && output_channel_ >= 64)*/)
					{
						conv1x1s1_sgemm_transform_kernel_neon();
					}
					else if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 1 && this->stride_w_ == 1)/* && (this->input_channel_ >= 16 && output_channel_ >= 16)*/&& this->output_channel_<512)
					{
						conv3x3s1_winograd64_transform_kernel_neon5();
					}
					else if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 2 && this->stride_w_ == 2))
					{
						conv3x3s2_transform_kernel_neon();
					}

					conv_im2col_sgemm_transform_kernel_neon();
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
					else if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 2 && this->stride_w_ == 2))
					{
						conv3x3s2_transform_kernel_int8_neon();
					}
					else if ((this->kernel_size_h_ == 1 && this->kernel_size_w_ == 1) && (this->stride_h_ == 2 && this->stride_w_ == 2))
					{
						conv1x1s1_sgemm_transform_kernel_int8_neon();
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
				if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 1 && this->stride_w_ == 1)&& this->output_channel_<512)
				{
					if (w <= 120 && h <= 120)
						conv3x3s1_winograd64_neon5(bottom_bordered, tops[0]);
					else
						conv3x3s1_neon(bottom_bordered, tops[0]);
				}
				else if ((this->kernel_size_h_ == 1 && this->kernel_size_w_ == 1) && (this->stride_h_ == 1 && this->stride_w_ == 1))
				{
					if (this->input_channel_ >= 64 && this->output_channel_ >= 64)
						conv1x1s1_sgemm_neon(bottom_bordered, tops[0]);
					else
						conv1x1s1_neon(bottom_bordered, tops[0]);
				}
				else if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 2 && this->stride_w_ == 2))
				{
					if (outw >= 8 && outh >= 8)
					{
						conv3x3s2_packed_neon(bottom_bordered, tops[0]);
					}
					else
					{
						conv_im2col_sgemm_neon(bottom_bordered, tops[0]);
					}
				}
				else
				{
					conv_im2col_sgemm_neon(bottom_bordered, tops[0]);
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			this->suffix_activation_cpu_f32(tops);
		}

		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv1x1s1_sgemm_transform_kernel_neon()
		{
			int inch = this->input_channel_;
			int outch = this->output_channel_;
			const float* kernel = this->weights_f32_[0]->cpu_data();
			// interleave
#if __ARM_NEON && __aarch64__
			kernel_tm_.reset(new memory::tensor<float>(std::vector<int>{1, outch / 8 + (outch % 8) / 4 + outch % 4, inch / 4 + inch % 4, 4 * 8}, -1, memory::NCHW, nullptr));
#else
			kernel_tm_.reset(new memory::tensor<float>(std::vector<int>{1, outch / 4 + outch % 4, inch / 4 + inch % 4, 4 * 4}, -1, memory::NCHW, nullptr));
#endif // __ARM_NEON && __aarch64__

			int p = 0;
#if __ARM_NEON && __aarch64__
			for (; p + 7 < outch; p += 8)
			{
				const float* kernel0 = kernel + (p + 0)*inch;
				const float* kernel1 = kernel + (p + 1)*inch;
				const float* kernel2 = kernel + (p + 2)*inch;
				const float* kernel3 = kernel + (p + 3)*inch;
				const float* kernel4 = kernel + (p + 4)*inch;
				const float* kernel5 = kernel + (p + 5)*inch;
				const float* kernel6 = kernel + (p + 6)*inch;
				const float* kernel7 = kernel + (p + 7)*inch;

				float* ktmp = kernel_tm_->mutable_cpu_data() + (p / 8) * kernel_tm_->width() * kernel_tm_->height();

				for (int q = 0; q < inch; q++)
				{
					// kernel0...7 0
					ktmp[0] = kernel0[0];
					ktmp[1] = kernel1[0];
					ktmp[2] = kernel2[0];
					ktmp[3] = kernel3[0];
					ktmp[4] = kernel4[0];
					ktmp[5] = kernel5[0];
					ktmp[6] = kernel6[0];
					ktmp[7] = kernel7[0];

					ktmp += 8;
					kernel0 += 1;
					kernel1 += 1;
					kernel2 += 1;
					kernel3 += 1;
					kernel4 += 1;
					kernel5 += 1;
					kernel6 += 1;
					kernel7 += 1;
				}
			}
#endif // __ARM_NEON && __aarch64__
			for (; p + 3 < outch; p += 4)
			{
				const float* kernel0 = kernel + (p + 0) * inch;
				const float* kernel1 = kernel + (p + 1) * inch;
				const float* kernel2 = kernel + (p + 2) * inch;
				const float* kernel3 = kernel + (p + 3) * inch;

#if __ARM_NEON && __aarch64__
				float* ktmp = kernel_tm_->mutable_cpu_data() + (p / 8 + (p % 8) / 4) * kernel_tm_->width() * kernel_tm_->height();
#else
				float* ktmp = kernel_tm_->mutable_cpu_data() + (p / 4) * kernel_tm_->width() * kernel_tm_->height();
#endif // __ARM_NEON && __aarch64__

				for (int q = 0; q < inch; q++)
				{
					// kernel0...3 0
					ktmp[0] = kernel0[0];
					ktmp[1] = kernel1[0];
					ktmp[2] = kernel2[0];
					ktmp[3] = kernel3[0];

					ktmp += 4;
					kernel0 += 1;
					kernel1 += 1;
					kernel2 += 1;
					kernel3 += 1;
				}
			}
			for (; p < outch; p++)
			{
				const float* kernel0 = kernel + p * inch;

#if __ARM_NEON && __aarch64__
				float* ktmp = kernel_tm_->mutable_cpu_data() + (p / 8 + (p % 8) / 4 + p % 4) * kernel_tm_->width() * kernel_tm_->height();
#else
				float* ktmp = kernel_tm_->mutable_cpu_data() + (p / 4 + p % 4) * kernel_tm_->width() * kernel_tm_->height();
#endif // __ARM_NEON && __aarch64__

				for (int q = 0; q < inch; q++)
				{
					ktmp[0] = kernel0[0];
					ktmp++;
					kernel0++;
				}
			}
		}

		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv3x3s1_winograd64_transform_kernel_neon5()
		{
			int inch = this->input_channel_;
			int outch = this->output_channel_;
			const float *kernel_data = this->weights_f32_[0]->cpu_data();

			kernel_tm_.reset(new memory::tensor<float>(std::vector<int>{1, outch, inch, 8 * 8}, -1, memory::NCHW));
			float *kernel_tm_data = kernel_tm_->mutable_cpu_data();
			int kernel_tm_w = kernel_tm_->width();
			int kernel_tm_h = kernel_tm_->height();
			int kernel_tm_cstep = kernel_tm_w * kernel_tm_h;

			const float ktm[8][3] = {
				{ 1.0f,     0.0f,     0.0f },
				{ -2.0f / 9,  -2.0f / 9,  -2.0f / 9 },
				{ -2.0f / 9,   2.0f / 9,  -2.0f / 9 },
				{ 1.0f / 90,  1.0f / 45,  2.0f / 45 },
				{ 1.0f / 90, -1.0f / 45,  2.0f / 45 },
				{ 1.0f / 45,  1.0f / 90, 1.0f / 180 },
				{ 1.0f / 45, -1.0f / 90, 1.0f / 180 },
				{ 0.0f,     0.0f,     1.0f }
			};

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
			for (int p = 0; p < outch; p++)
			{
				for (int q = 0; q < inch; q++)
				{
					const float* kernel0 = kernel_data + p * inch * 9 + q * 9;
					float* kernel_tm0 = kernel_tm_data + p * kernel_tm_cstep + q * kernel_tm_w;

					// transform kernel, transposed
					const float* k0 = kernel0;
					const float* k1 = kernel0 + 3;
					const float* k2 = kernel0 + 6;

					// h
					float tmp[8][3];
					for (int i = 0; i < 8; i++)
					{
						tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
						tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
						tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
					}

					// v
					for (int j = 0; j < 8; j++)
					{
						float* tmpp = &tmp[j][0];

						for (int i = 0; i < 8; i++)
						{
							kernel_tm0[j * 8 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
						}
					}
				}
			}


			// optimized layout for winograd5
			// interleave weights
			//     Mat kernel_tm2(8*8, inch, outch);
			//     Mat kernel_tm2(inch, 64, outch);
#if __ARM_NEON && __aarch64__
			memory::tensor<float> kernel_tm2(std::vector<int>{1, outch / 8 + (outch % 8) / 4 + outch % 4, 64, 8 * 4 * (inch / 4) + 8 * (inch % 4)}, -1, memory::NCHW);
#else
			memory::tensor<float> kernel_tm2(std::vector<int>{1, outch / 4 + outch % 4, 64, 4 * 4 * (inch / 4) + 4 * (inch % 4)}, -1, memory::NCHW);
#endif
			float *kernel_tm2_data = kernel_tm2.mutable_cpu_data();
			int kernel_tm2_w = kernel_tm2.width();
			int kernel_tm2_h = kernel_tm2.height();
			int kernel_tm2_cstep = kernel_tm2_w * kernel_tm2_h;

			int p = 0;
#if __aarch64__
			for (; p + 7 < outch; p += 8)
			{
				const float *kernel0_tm = kernel_tm_data + (p + 0) * kernel_tm_cstep;
				const float *kernel1_tm = kernel_tm_data + (p + 1) * kernel_tm_cstep;
				const float *kernel2_tm = kernel_tm_data + (p + 2) * kernel_tm_cstep;
				const float *kernel3_tm = kernel_tm_data + (p + 3) * kernel_tm_cstep;
				const float *kernel4_tm = kernel_tm_data + (p + 4) * kernel_tm_cstep;
				const float *kernel5_tm = kernel_tm_data + (p + 5) * kernel_tm_cstep;
				const float *kernel6_tm = kernel_tm_data + (p + 6) * kernel_tm_cstep;
				const float *kernel7_tm = kernel_tm_data + (p + 7) * kernel_tm_cstep;

				float *ktm2 = kernel_tm2_data + (p / 8) * kernel_tm2_cstep;

				for (int r = 0; r < 64; r++)
				{
					float* ktm2p = ktm2 + (r)* kernel_tm2_w;

					for (int q = 0; q < inch; q++)
					{
						const float* ktm0_0 = kernel0_tm + (q)* kernel_tm_w;
						const float* ktm1_0 = kernel1_tm + (q)* kernel_tm_w;
						const float* ktm2_0 = kernel2_tm + (q)* kernel_tm_w;
						const float* ktm3_0 = kernel3_tm + (q)* kernel_tm_w;
						const float* ktm4_0 = kernel4_tm + (q)* kernel_tm_w;
						const float* ktm5_0 = kernel5_tm + (q)* kernel_tm_w;
						const float* ktm6_0 = kernel6_tm + (q)* kernel_tm_w;
						const float* ktm7_0 = kernel7_tm + (q)* kernel_tm_w;

						ktm2p[0] = ktm0_0[r];
						ktm2p[1] = ktm1_0[r];
						ktm2p[2] = ktm2_0[r];
						ktm2p[3] = ktm3_0[r];
						ktm2p[4] = ktm4_0[r];
						ktm2p[5] = ktm5_0[r];
						ktm2p[6] = ktm6_0[r];
						ktm2p[7] = ktm7_0[r];

						ktm2p += 8;
					}
				}
			}
#endif // __aarch64__
			for (; p + 3 < outch; p += 4)
			{
				const float* kernel0_tm = kernel_tm_data + (p)* kernel_tm_cstep;
				const float* kernel1_tm = kernel_tm_data + (p + 1) * kernel_tm_cstep;
				const float* kernel2_tm = kernel_tm_data + (p + 2) * kernel_tm_cstep;
				const float* kernel3_tm = kernel_tm_data + (p + 3) * kernel_tm_cstep;

#if __ARM_NEON && __aarch64__
				float *ktm2 = kernel_tm2_data + (p / 8 + (p % 8) / 4) * kernel_tm2_cstep;
#else
				float *ktm2 = kernel_tm2_data + (p / 4) * kernel_tm2_cstep;
#endif

				for (int r = 0; r < 64; r++)
				{
					float* ktm2p = ktm2 + (r)* kernel_tm2_w;

					for (int q = 0; q < inch; q++)
					{
						const float* ktm0_0 = kernel0_tm + (q)* kernel_tm_w;
						const float* ktm1_0 = kernel1_tm + (q)* kernel_tm_w;
						const float* ktm2_0 = kernel2_tm + (q)* kernel_tm_w;
						const float* ktm3_0 = kernel3_tm + (q)* kernel_tm_w;

						ktm2p[0] = ktm0_0[r];
						ktm2p[1] = ktm1_0[r];
						ktm2p[2] = ktm2_0[r];
						ktm2p[3] = ktm3_0[r];

						ktm2p += 4;
					}
				}
			}
			for (; p < outch; p++)
			{
				const float *kernel0_tm = kernel_tm_data + (p)* kernel_tm_cstep;

#if __ARM_NEON && __aarch64__
				float * ktm2 = kernel_tm2_data + (p / 8 + (p % 8) / 4 + p % 4) * kernel_tm2_cstep;
#else
				float * ktm2 = kernel_tm2_data + (p / 4 + p % 4) * kernel_tm2_cstep;
#endif

				for (int r = 0; r < 64; r++)
				{
					float* ktm2p = ktm2 + (r)* kernel_tm2_w;

					for (int q = 0; q < inch; q++)
					{
						const float* ktm0_0 = kernel0_tm + (q)* kernel_tm_w;

						ktm2p[0] = ktm0_0[r];

						ktm2p += 1;
					}
				}
			}

			*kernel_tm_ = kernel_tm2;
		}

		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv3x3s2_transform_kernel_neon()
		{
			int inch = this->input_channel_;
			int outch = this->output_channel_;
			kernel_tm_.reset(new memory::tensor<float>(std::vector<int>{1, outch / 8 + outch % 8, inch, 8 * 9}, -1, memory::NCHW));

			const float* kernel = this->weights_f32_[0]->cpu_data();

			int p = 0;
			for (; p + 7 < outch; p += 8)
			{
				const float* k0 = kernel + (p + 0)*inch * 9;
				const float* k1 = kernel + (p + 1)*inch * 9;
				const float* k2 = kernel + (p + 2)*inch * 9;
				const float* k3 = kernel + (p + 3)*inch * 9;
				const float* k4 = kernel + (p + 4)*inch * 9;
				const float* k5 = kernel + (p + 5)*inch * 9;
				const float* k6 = kernel + (p + 6)*inch * 9;
				const float* k7 = kernel + (p + 7)*inch * 9;

				float* ktmp = kernel_tm_->mutable_cpu_data() + (p / 8) * kernel_tm_->width() * kernel_tm_->height();

				for (int q = 0; q < inch; q++)
				{
					for (int k = 0; k < 9; k++)
					{
						ktmp[0] = k0[k];
						ktmp[1] = k1[k];
						ktmp[2] = k2[k];
						ktmp[3] = k3[k];
						ktmp[4] = k4[k];
						ktmp[5] = k5[k];
						ktmp[6] = k6[k];
						ktmp[7] = k7[k];
						ktmp += 8;
					}

					k0 += 9;
					k1 += 9;
					k2 += 9;
					k3 += 9;
					k4 += 9;
					k5 += 9;
					k6 += 9;
					k7 += 9;
				}
			}
			for (; p < outch; p++)
			{
				const float* k0 = kernel + (p + 0)*inch * 9;

				float* ktmp = kernel_tm_->mutable_cpu_data() + (p / 8 + p % 8) * kernel_tm_->width() * kernel_tm_->height();

				for (int q = 0; q < inch; q++)
				{
					for (int k = 0; k < 9; k++)
					{
						ktmp[k] = k0[k];
					}
					ktmp += 9;

					k0 += 9;
				}
			}
		}

		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv_im2col_sgemm_transform_kernel_neon()
		{
			int inch = this->input_channel_;
			int outch = this->output_channel_;
			int kernel_size = this->kernel_size_w_ * this->kernel_size_h_;
			const float* kernel = this->weights_f32_[0]->cpu_data();

#if __ARM_NEON && __aarch64__
			// kernel memory packed 8 x 8
			kernel_tm_gemm_.reset(new memory::tensor<float>(std::vector<int>{ 1, outch / 8 + (outch % 8) / 4 + outch % 4, inch, 8 * kernel_size}, -1, memory::NCHW));
#else    
			// kernel memory packed 4 x 8
			kernel_tm_gemm_.reset(new memory::tensor<float>(std::vector<int>{1, outch / 4 + outch % 4, inch, 4 * kernel_size}, -1, memory::NCHW));
#endif
			float *kernel_tm_gemm_data = kernel_tm_gemm_->mutable_cpu_data();
			int kernel_tm_gemm_cstep = kernel_tm_gemm_->width() * kernel_tm_gemm_->height();
			int nn_outch = 0;
			int remain_outch_start = 0;

#if __ARM_NEON && __aarch64__
			nn_outch = outch >> 3;
			remain_outch_start = nn_outch << 3;

			for (int pp = 0; pp < nn_outch; pp++)
			{
				int p = pp * 8;

				const float* k0 = kernel + (p + 0)*inch*kernel_size;
				const float* k1 = kernel + (p + 1)*inch*kernel_size;
				const float* k2 = kernel + (p + 2)*inch*kernel_size;
				const float* k3 = kernel + (p + 3)*inch*kernel_size;
				const float* k4 = kernel + (p + 4)*inch*kernel_size;
				const float* k5 = kernel + (p + 5)*inch*kernel_size;
				const float* k6 = kernel + (p + 6)*inch*kernel_size;
				const float* k7 = kernel + (p + 7)*inch*kernel_size;

				float* ktmp = kernel_tm_gemm_data + (p / 8) * kernel_tm_gemm_cstep;

				for (int q = 0; q < inch*kernel_size; q++)
				{
					ktmp[0] = k0[0];
					ktmp[1] = k1[0];
					ktmp[2] = k2[0];
					ktmp[3] = k3[0];
					ktmp[4] = k4[0];
					ktmp[5] = k5[0];
					ktmp[6] = k6[0];
					ktmp[7] = k7[0];
					ktmp += 8;

					k0 += 1;
					k1 += 1;
					k2 += 1;
					k3 += 1;
					k4 += 1;
					k5 += 1;
					k6 += 1;
					k7 += 1;
				}
			}
#endif

			nn_outch = (outch - remain_outch_start) >> 2;

			for (int pp = 0; pp < nn_outch; pp++)
			{
				int p = remain_outch_start + pp * 4;

				const float* k0 = kernel + (p + 0)*inch*kernel_size;
				const float* k1 = kernel + (p + 1)*inch*kernel_size;
				const float* k2 = kernel + (p + 2)*inch*kernel_size;
				const float* k3 = kernel + (p + 3)*inch*kernel_size;

#if __ARM_NEON && __aarch64__
				float* ktmp = kernel_tm_gemm_data + (p / 8 + (p % 8) / 4) * kernel_tm_gemm_cstep;
#else
				float* ktmp = kernel_tm_gemm_data + (p / 4) * kernel_tm_gemm_cstep;
#endif // __ARM_NEON && __aarch64__

				for (int q = 0; q < inch*kernel_size; q++)
				{
					ktmp[0] = k0[0];
					ktmp[1] = k1[0];
					ktmp[2] = k2[0];
					ktmp[3] = k3[0];
					ktmp += 4;

					k0 += 1;
					k1 += 1;
					k2 += 1;
					k3 += 1;
				}
			}

			remain_outch_start += nn_outch << 2;

			for (int p = remain_outch_start; p < outch; p++)
			{
				const float* k0 = kernel + (p + 0)*inch*kernel_size;

#if __ARM_NEON && __aarch64__
				float* ktmp = kernel_tm_gemm_data + (p / 8 + (p % 8) / 4 + p % 4) * kernel_tm_gemm_cstep;
#else
				float* ktmp = kernel_tm_gemm_data + (p / 4 + p % 4) * kernel_tm_gemm_cstep;
#endif // __ARM_NEON && __aarch64__            

				for (int q = 0; q < inch*kernel_size; q++)
				{
					ktmp[0] = k0[0];
					ktmp++;
					k0++;
				}
			}
		}

		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv1x1s1_sgemm_neon(const std::shared_ptr < memory::tensor<float>>& bottom, std::shared_ptr < memory::tensor<float>>& top)
		{
			int num = bottom->num();
			int w = bottom->width();
			int h = bottom->height();
			int inch = bottom->channels();
			int outch = top->channels();

			const int size = w * h;
			const float* bias = nullptr;
			if (this->bias_term_)
				bias = this->weights_f32_[1]->cpu_data();

			int bottom_cstep = w * h;
			int top_cstep = top->width() * top->height();
			const float *kernel_data = kernel_tm_->cpu_data();
			int kernel_cstep = kernel_tm_->width() * kernel_tm_->height();

			// interleave
			memory::tensor<float> tmp(std::vector<int>{1, size / 8 + (size % 8) / 4 + size % 4, inch / 4 + inch % 4, 8 * 4}, -1, memory::NCHW, nullptr);
			float *tmp_data = tmp.mutable_cpu_data();
			int tmp_cstep = tmp.width() * tmp.height();

			for (int num_i = 0; num_i < num; num_i++)
			{
				const float *bottom_data = bottom->cpu_data() + num_i * inch * bottom_cstep;
				float *top_data = top->mutable_cpu_data() + num_i * outch * top_cstep;

				{
					int nn_size = size >> 3;
					int remain_size_start = nn_size << 3;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
					for (int ii = 0; ii < nn_size; ii++)
					{
						int i = ii * 8;

						const float* img0 = bottom_data;
						img0 += i;

						float* tmpptr = tmp_data + (i / 8) * tmp_cstep;

						for (int q = 0; q < inch; q++)
						{
#if __ARM_NEON
#if __aarch64__
							vst1q_f32(tmpptr, vld1q_f32(img0));
							vst1q_f32(tmpptr + 4, vld1q_f32(img0 + 4));

							tmpptr += 8;
							img0 += bottom_cstep;
#else
							asm volatile(
								"pld        [%0, #256]          \n"
								"vld1.f32   {d0-d3}, [%0 :128]  \n"
								"vst1.f32   {d0-d3}, [%1 :128]! \n"
								: "=r"(img0),   // %0
								"=r"(tmpptr)  // %1
								: "0"(img0),
								"1"(tmpptr)
								: "memory", "q0", "q1"
								);

							img0 += bottom_cstep;
#endif // __aarch64__
#else
							tmpptr[0] = img0[0];
							tmpptr[1] = img0[1];
							tmpptr[2] = img0[2];
							tmpptr[3] = img0[3];
							tmpptr[4] = img0[4];
							tmpptr[5] = img0[5];
							tmpptr[6] = img0[6];
							tmpptr[7] = img0[7];
							tmpptr += 8;
							img0 += bottom_cstep;

#endif // __ARM_NEON
						}
					}

					nn_size = (size - remain_size_start) >> 2;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
					for (int ii = 0; ii < nn_size; ii++)
					{
						int i = remain_size_start + ii * 4;

						const float* img0 = bottom_data;
						img0 += i;

						float* tmpptr = tmp_data + (i / 8 + (i % 8) / 4) * tmp_cstep;

						for (int q = 0; q < inch; q++)
						{
#if __ARM_NEON
#if __aarch64__
							vst1q_f32(tmpptr, vld1q_f32(img0));

							tmpptr += 4;
							img0 += bottom_cstep;
#else
							asm volatile(
								"pld        [%0, #128]          \n"
								"vld1.f32   {d0-d1}, [%0 :128]  \n"
								"vst1.f32   {d0-d1}, [%1 :128]! \n"
								: "=r"(img0),   // %0
								"=r"(tmpptr)  // %1
								: "0"(img0),
								"1"(tmpptr)
								: "memory", "q0"
								);

							img0 += bottom_cstep;
#endif // __aarch64__
#else
							tmpptr[0] = img0[0];
							tmpptr[1] = img0[1];
							tmpptr[2] = img0[2];
							tmpptr[3] = img0[3];
							tmpptr += 4;
							img0 += bottom_cstep;

#endif // __ARM_NEON
						}
					}

					remain_size_start += nn_size << 2;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
					for (int i = remain_size_start; i < size; i++)
					{
						const float* img0 = bottom_data;
						img0 += i;

						float* tmpptr = tmp_data + (i / 8 + (i % 8) / 4 + i % 4) * tmp_cstep;

						for (int q = 0; q < inch; q++)
						{
							tmpptr[0] = img0[0];
							tmpptr++;
							img0 += bottom_cstep;
						}
					}
				}

				int nn_outch = 0;
				int remain_outch_start = 0;

#if __ARM_NEON && __aarch64__
				nn_outch = outch >> 3;
				remain_outch_start = nn_outch << 3;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
				for (int pp = 0; pp < nn_outch; pp++)
				{
					int p = pp * 8;

					float* outptr0 = top_data + (p + 0) * top_cstep;
					float* outptr1 = top_data + (p + 1) * top_cstep;
					float* outptr2 = top_data + (p + 2) * top_cstep;
					float* outptr3 = top_data + (p + 3) * top_cstep;
					float* outptr4 = top_data + (p + 4) * top_cstep;
					float* outptr5 = top_data + (p + 5) * top_cstep;
					float* outptr6 = top_data + (p + 6) * top_cstep;
					float* outptr7 = top_data + (p + 7) * top_cstep;

					const float zeros[8] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
					const float* biasptr = this->bias_term_ ? bias + p : zeros;

					int i = 0;

					for (; i + 7 < size; i += 8)
					{
						const float* tmpptr = tmp_data + (i / 8) * tmp_cstep;
						const float* kptr = kernel_data + (p / 8) * kernel_cstep;

						asm volatile(
							"ld1    {v0.4s, v1.4s}, [%20]   \n"
							"dup    v16.4s, v0.s[0]         \n"
							"dup    v17.4s, v0.s[0]         \n"
							"dup    v18.4s, v0.s[1]         \n"
							"dup    v19.4s, v0.s[1]         \n"
							"dup    v20.4s, v0.s[2]         \n"
							"dup    v21.4s, v0.s[2]         \n"
							"dup    v22.4s, v0.s[3]         \n"
							"dup    v23.4s, v0.s[3]         \n"
							"dup    v24.4s, v1.s[0]         \n"
							"dup    v25.4s, v1.s[0]         \n"
							"dup    v26.4s, v1.s[1]         \n"
							"dup    v27.4s, v1.s[1]         \n"
							"dup    v28.4s, v1.s[2]         \n"
							"dup    v29.4s, v1.s[2]         \n"
							"dup    v30.4s, v1.s[3]         \n"
							"dup    v31.4s, v1.s[3]         \n"

							// inch loop
							"lsr    w4, %w21, #2            \n"// w4 = nn = inch >> 2
							"cmp    w4, #0                  \n"
							"beq    1f                      \n"

							"0:                             \n"

							"prfm   pldl1keep, [%8, #512]   \n"
							"ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%8], #64   \n"

							"prfm   pldl1keep, [%9, #512]   \n"
							"ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64     \n"

							"fmla   v16.4s, v8.4s, v0.s[0]  \n"
							"fmla   v18.4s, v8.4s, v0.s[1]  \n"
							"fmla   v20.4s, v8.4s, v0.s[2]  \n"
							"fmla   v22.4s, v8.4s, v0.s[3]  \n"

							"fmla   v17.4s, v9.4s, v0.s[0]  \n"
							"fmla   v19.4s, v9.4s, v0.s[1]  \n"
							"fmla   v21.4s, v9.4s, v0.s[2]  \n"
							"fmla   v23.4s, v9.4s, v0.s[3]  \n"

							"fmla   v24.4s, v8.4s, v1.s[0]  \n"
							"fmla   v26.4s, v8.4s, v1.s[1]  \n"
							"fmla   v28.4s, v8.4s, v1.s[2]  \n"
							"fmla   v30.4s, v8.4s, v1.s[3]  \n"

							"fmla   v25.4s, v9.4s, v1.s[0]  \n"
							"fmla   v27.4s, v9.4s, v1.s[1]  \n"
							"fmla   v29.4s, v9.4s, v1.s[2]  \n"
							"fmla   v31.4s, v9.4s, v1.s[3]  \n"

							"prfm   pldl1keep, [%8, #512]   \n"
							"ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%8], #64 \n"

							"fmla   v16.4s, v10.4s, v2.s[0] \n"
							"fmla   v18.4s, v10.4s, v2.s[1] \n"
							"fmla   v20.4s, v10.4s, v2.s[2] \n"
							"fmla   v22.4s, v10.4s, v2.s[3] \n"

							"fmla   v17.4s, v11.4s, v2.s[0] \n"
							"fmla   v19.4s, v11.4s, v2.s[1] \n"
							"fmla   v21.4s, v11.4s, v2.s[2] \n"
							"fmla   v23.4s, v11.4s, v2.s[3] \n"

							"fmla   v24.4s, v10.4s, v3.s[0] \n"
							"fmla   v26.4s, v10.4s, v3.s[1] \n"
							"fmla   v28.4s, v10.4s, v3.s[2] \n"
							"fmla   v30.4s, v10.4s, v3.s[3] \n"

							"fmla   v25.4s, v11.4s, v3.s[0] \n"
							"fmla   v27.4s, v11.4s, v3.s[1] \n"
							"fmla   v29.4s, v11.4s, v3.s[2] \n"
							"fmla   v31.4s, v11.4s, v3.s[3] \n"

							"prfm   pldl1keep, [%9, #512]   \n"
							"ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%9], #64     \n"

							"fmla   v16.4s, v12.4s, v4.s[0] \n"
							"fmla   v18.4s, v12.4s, v4.s[1] \n"
							"fmla   v20.4s, v12.4s, v4.s[2] \n"
							"fmla   v22.4s, v12.4s, v4.s[3] \n"

							"fmla   v17.4s, v13.4s, v4.s[0] \n"
							"fmla   v19.4s, v13.4s, v4.s[1] \n"
							"fmla   v21.4s, v13.4s, v4.s[2] \n"
							"fmla   v23.4s, v13.4s, v4.s[3] \n"

							"fmla   v24.4s, v12.4s, v5.s[0] \n"
							"fmla   v26.4s, v12.4s, v5.s[1] \n"
							"fmla   v28.4s, v12.4s, v5.s[2] \n"
							"fmla   v30.4s, v12.4s, v5.s[3] \n"

							"fmla   v25.4s, v13.4s, v5.s[0] \n"
							"fmla   v27.4s, v13.4s, v5.s[1] \n"
							"fmla   v29.4s, v13.4s, v5.s[2] \n"
							"fmla   v31.4s, v13.4s, v5.s[3] \n"

							"subs   w4, w4, #1              \n"

							"fmla   v16.4s, v14.4s, v6.s[0] \n"
							"fmla   v18.4s, v14.4s, v6.s[1] \n"
							"fmla   v20.4s, v14.4s, v6.s[2] \n"
							"fmla   v22.4s, v14.4s, v6.s[3] \n"

							"fmla   v17.4s, v15.4s, v6.s[0] \n"
							"fmla   v19.4s, v15.4s, v6.s[1] \n"
							"fmla   v21.4s, v15.4s, v6.s[2] \n"
							"fmla   v23.4s, v15.4s, v6.s[3] \n"

							"fmla   v24.4s, v14.4s, v7.s[0] \n"
							"fmla   v26.4s, v14.4s, v7.s[1] \n"
							"fmla   v28.4s, v14.4s, v7.s[2] \n"
							"fmla   v30.4s, v14.4s, v7.s[3] \n"

							"fmla   v25.4s, v15.4s, v7.s[0] \n"
							"fmla   v27.4s, v15.4s, v7.s[1] \n"
							"fmla   v29.4s, v15.4s, v7.s[2] \n"
							"fmla   v31.4s, v15.4s, v7.s[3] \n"

							"bne    0b                      \n"

							"1:                             \n"

							// remain loop
							"and    w4, %w21, #3            \n"// w4 = remain = inch & 3;
							"cmp    w4, #0                  \n"
							"beq    3f                      \n"

							"2:                             \n"

							"prfm   pldl1keep, [%8, #256]   \n"
							"ld1    {v8.4s, v9.4s}, [%8], #32   \n"

							"prfm   pldl1keep, [%9, #256]   \n"
							"ld1    {v0.4s, v1.4s}, [%9], #32   \n"

							"fmla   v16.4s, v8.4s, v0.s[0]  \n"
							"fmla   v18.4s, v8.4s, v0.s[1]  \n"
							"fmla   v20.4s, v8.4s, v0.s[2]  \n"
							"fmla   v22.4s, v8.4s, v0.s[3]  \n"

							"fmla   v17.4s, v9.4s, v0.s[0]  \n"
							"fmla   v19.4s, v9.4s, v0.s[1]  \n"
							"fmla   v21.4s, v9.4s, v0.s[2]  \n"
							"fmla   v23.4s, v9.4s, v0.s[3]  \n"

							"subs   w4, w4, #1              \n"

							"fmla   v24.4s, v8.4s, v1.s[0]  \n"
							"fmla   v26.4s, v8.4s, v1.s[1]  \n"
							"fmla   v28.4s, v8.4s, v1.s[2]  \n"
							"fmla   v30.4s, v8.4s, v1.s[3]  \n"

							"fmla   v25.4s, v9.4s, v1.s[0]  \n"
							"fmla   v27.4s, v9.4s, v1.s[1]  \n"
							"fmla   v29.4s, v9.4s, v1.s[2]  \n"
							"fmla   v31.4s, v9.4s, v1.s[3]  \n"

							"bne    2b                      \n"

							"3:                             \n"

							"st1    {v16.4s, v17.4s}, [%0], #32 \n"
							"st1    {v18.4s, v19.4s}, [%1], #32 \n"
							"st1    {v20.4s, v21.4s}, [%2], #32 \n"
							"st1    {v22.4s, v23.4s}, [%3], #32 \n"
							"st1    {v24.4s, v25.4s}, [%4], #32 \n"
							"st1    {v26.4s, v27.4s}, [%5], #32 \n"
							"st1    {v28.4s, v29.4s}, [%6], #32 \n"
							"st1    {v30.4s, v31.4s}, [%7], #32 \n"

							: "=r"(outptr0),    // %0
							"=r"(outptr1),    // %1
							"=r"(outptr2),    // %2
							"=r"(outptr3),    // %3
							"=r"(outptr4),    // %4
							"=r"(outptr5),    // %5
							"=r"(outptr6),    // %6
							"=r"(outptr7),    // %7
							"=r"(tmpptr),     // %8
							"=r"(kptr)        // %9
							: "0"(outptr0),
							"1"(outptr1),
							"2"(outptr2),
							"3"(outptr3),
							"4"(outptr4),
							"5"(outptr5),
							"6"(outptr6),
							"7"(outptr7),
							"8"(tmpptr),
							"9"(kptr),
							"r"(biasptr),     // %20
							"r"(inch)         // %21
							: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
							);
					}

					for (; i + 3 < size; i += 4)
					{
						const float* tmpptr = tmp_data + (i / 8 + (i % 8) / 4) * tmp_cstep;
						const float* kptr = kernel_data + (p / 8) * kernel_cstep;

						asm volatile(
							"ld1    {v0.4s, v1.4s}, [%20]   \n"
							"dup    v16.4s, v0.s[0]         \n"
							"dup    v17.4s, v0.s[1]         \n"
							"dup    v18.4s, v0.s[2]         \n"
							"dup    v19.4s, v0.s[3]         \n"
							"dup    v20.4s, v1.s[0]         \n"
							"dup    v21.4s, v1.s[1]         \n"
							"dup    v22.4s, v1.s[2]         \n"
							"dup    v23.4s, v1.s[3]         \n"

							// inch loop
							"lsr    w4, %w21, #2            \n"// w4 = nn = inch >> 2
							"cmp    w4, #0                  \n"
							"beq    1f                      \n"

							"0:                             \n"

							"prfm   pldl1keep, [%8, #512]   \n"
							"ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%8], #64   \n"

							"prfm   pldl1keep, [%9, #512]   \n"
							"ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64     \n"

							"fmla   v16.4s, v8.4s, v0.s[0]  \n"
							"fmla   v17.4s, v8.4s, v0.s[1]  \n"
							"fmla   v18.4s, v8.4s, v0.s[2]  \n"
							"fmla   v19.4s, v8.4s, v0.s[3]  \n"
							"fmla   v20.4s, v8.4s, v1.s[0]  \n"
							"fmla   v21.4s, v8.4s, v1.s[1]  \n"
							"fmla   v22.4s, v8.4s, v1.s[2]  \n"
							"fmla   v23.4s, v8.4s, v1.s[3]  \n"

							"prfm   pldl1keep, [%9, #512]   \n"
							"ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%9], #64     \n"

							"fmla   v16.4s, v9.4s, v2.s[0]  \n"
							"fmla   v17.4s, v9.4s, v2.s[1]  \n"
							"fmla   v18.4s, v9.4s, v2.s[2]  \n"
							"fmla   v19.4s, v9.4s, v2.s[3]  \n"
							"fmla   v20.4s, v9.4s, v3.s[0]  \n"
							"fmla   v21.4s, v9.4s, v3.s[1]  \n"
							"fmla   v22.4s, v9.4s, v3.s[2]  \n"
							"fmla   v23.4s, v9.4s, v3.s[3]  \n"

							"subs   w4, w4, #1              \n"

							"fmla   v16.4s, v10.4s, v4.s[0] \n"
							"fmla   v17.4s, v10.4s, v4.s[1] \n"
							"fmla   v18.4s, v10.4s, v4.s[2] \n"
							"fmla   v19.4s, v10.4s, v4.s[3] \n"
							"fmla   v20.4s, v10.4s, v5.s[0] \n"
							"fmla   v21.4s, v10.4s, v5.s[1] \n"
							"fmla   v22.4s, v10.4s, v5.s[2] \n"
							"fmla   v23.4s, v10.4s, v5.s[3] \n"

							"fmla   v16.4s, v11.4s, v6.s[0] \n"
							"fmla   v17.4s, v11.4s, v6.s[1] \n"
							"fmla   v18.4s, v11.4s, v6.s[2] \n"
							"fmla   v19.4s, v11.4s, v6.s[3] \n"
							"fmla   v20.4s, v11.4s, v7.s[0] \n"
							"fmla   v21.4s, v11.4s, v7.s[1] \n"
							"fmla   v22.4s, v11.4s, v7.s[2] \n"
							"fmla   v23.4s, v11.4s, v7.s[3] \n"

							"bne    0b                      \n"

							"1:                             \n"

							// remain loop
							"and    w4, %w21, #3            \n"// w4 = remain = inch & 3;
							"cmp    w4, #0                  \n"
							"beq    3f                      \n"

							"2:                             \n"

							"prfm   pldl1keep, [%8, #128]   \n"
							"ld1    {v8.4s}, [%8], #16      \n"

							"prfm   pldl1keep, [%9, #256]   \n"
							"ld1    {v0.4s, v1.4s}, [%9], #32   \n"

							"fmla   v16.4s, v8.4s, v0.s[0]  \n"
							"fmla   v17.4s, v8.4s, v0.s[1]  \n"
							"fmla   v18.4s, v8.4s, v0.s[2]  \n"
							"fmla   v19.4s, v8.4s, v0.s[3]  \n"

							"subs   w4, w4, #1              \n"

							"fmla   v20.4s, v8.4s, v1.s[0]  \n"
							"fmla   v21.4s, v8.4s, v1.s[1]  \n"
							"fmla   v22.4s, v8.4s, v1.s[2]  \n"
							"fmla   v23.4s, v8.4s, v1.s[3]  \n"

							"bne    2b                      \n"

							"3:                             \n"

							"st1    {v16.4s}, [%0], #16     \n"
							"st1    {v17.4s}, [%1], #16     \n"
							"st1    {v18.4s}, [%2], #16     \n"
							"st1    {v19.4s}, [%3], #16     \n"
							"st1    {v20.4s}, [%4], #16     \n"
							"st1    {v21.4s}, [%5], #16     \n"
							"st1    {v22.4s}, [%6], #16     \n"
							"st1    {v23.4s}, [%7], #16     \n"

							: "=r"(outptr0),    // %0
							"=r"(outptr1),    // %1
							"=r"(outptr2),    // %2
							"=r"(outptr3),    // %3
							"=r"(outptr4),    // %4
							"=r"(outptr5),    // %5
							"=r"(outptr6),    // %6
							"=r"(outptr7),    // %7
							"=r"(tmpptr),     // %8
							"=r"(kptr)        // %9
							: "0"(outptr0),
							"1"(outptr1),
							"2"(outptr2),
							"3"(outptr3),
							"4"(outptr4),
							"5"(outptr5),
							"6"(outptr6),
							"7"(outptr7),
							"8"(tmpptr),
							"9"(kptr),
							"r"(biasptr),     // %20
							"r"(inch)         // %21
							: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
							);
					}

					for (; i < size; i++)
					{
						const float* tmpptr = tmp_data + (i / 8 + (i % 8) / 4 + i % 4) * tmp_cstep;
						const float* kptr = kernel_data + (p / 8) * kernel_cstep;

						asm volatile(
							"ld1    {v24.4s, v25.4s}, [%20] \n"

							// inch loop
							"lsr    w4, %w21, #2            \n"// w4 = nn = inch >> 2
							"cmp    w4, #0                  \n"
							"beq    1f                      \n"

							"eor    v16.16b, v16.16b, v16.16b  \n"
							"eor    v17.16b, v17.16b, v17.16b  \n"
							"eor    v18.16b, v18.16b, v18.16b  \n"
							"eor    v19.16b, v19.16b, v19.16b  \n"
							"eor    v20.16b, v20.16b, v20.16b  \n"
							"eor    v21.16b, v21.16b, v21.16b  \n"
							"eor    v22.16b, v22.16b, v22.16b  \n"
							"eor    v23.16b, v23.16b, v23.16b  \n"

							"0:                             \n"

							"prfm   pldl1keep, [%8, #128]   \n"
							"ld1    {v8.4s}, [%8], #16      \n"

							"prfm   pldl1keep, [%9, #512]   \n"
							"ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64     \n"

							"fmla   v16.4s, v0.4s, v8.s[0]  \n"
							"fmla   v17.4s, v1.4s, v8.s[0]  \n"
							"fmla   v18.4s, v2.4s, v8.s[1]  \n"
							"fmla   v19.4s, v3.4s, v8.s[1]  \n"

							"prfm   pldl1keep, [%9, #512]   \n"
							"ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%9], #64     \n"

							"subs   w4, w4, #1              \n"

							"fmla   v20.4s, v4.4s, v8.s[2]  \n"
							"fmla   v21.4s, v5.4s, v8.s[2]  \n"
							"fmla   v22.4s, v6.4s, v8.s[3]  \n"
							"fmla   v23.4s, v7.4s, v8.s[3]  \n"

							"bne    0b                      \n"

							"fadd   v16.4s, v16.4s, v18.4s  \n"
							"fadd   v17.4s, v17.4s, v19.4s  \n"
							"fadd   v20.4s, v20.4s, v22.4s  \n"
							"fadd   v21.4s, v21.4s, v23.4s  \n"
							"fadd   v16.4s, v16.4s, v20.4s  \n"
							"fadd   v17.4s, v17.4s, v21.4s  \n"
							"fadd   v24.4s, v24.4s, v16.4s  \n"
							"fadd   v25.4s, v25.4s, v17.4s  \n"

							"1:                             \n"

							// remain loop
							"and    w4, %w21, #3            \n"// w4 = remain = inch & 3;
							"cmp    w4, #0                  \n"
							"beq    3f                      \n"

							"2:                             \n"

							"prfm   pldl1keep, [%8, #32]    \n"
							"ld1r   {v8.4s}, [%8], #4       \n"

							"prfm   pldl1keep, [%9, #256]   \n"
							"ld1    {v0.4s, v1.4s}, [%9], #32   \n"

							"subs   w4, w4, #1              \n"

							"fmla   v24.4s, v8.4s, v0.4s    \n"
							"fmla   v25.4s, v8.4s, v1.4s    \n"

							"bne    2b                      \n"

							"3:                             \n"

							"st1    {v24.s}[0],[%0], #4     \n"
							"st1    {v24.s}[1],[%1], #4     \n"
							"st1    {v24.s}[2],[%2], #4     \n"
							"st1    {v24.s}[3],[%3], #4     \n"
							"st1    {v25.s}[0],[%4], #4     \n"
							"st1    {v25.s}[1],[%5], #4     \n"
							"st1    {v25.s}[2],[%6], #4     \n"
							"st1    {v25.s}[3],[%7], #4     \n"

							: "=r"(outptr0),    // %0
							"=r"(outptr1),    // %1
							"=r"(outptr2),    // %2
							"=r"(outptr3),    // %3
							"=r"(outptr4),    // %4
							"=r"(outptr5),    // %5
							"=r"(outptr6),    // %6
							"=r"(outptr7),    // %7
							"=r"(tmpptr),     // %8
							"=r"(kptr)        // %9
							: "0"(outptr0),
							"1"(outptr1),
							"2"(outptr2),
							"3"(outptr3),
							"4"(outptr4),
							"5"(outptr5),
							"6"(outptr6),
							"7"(outptr7),
							"8"(tmpptr),
							"9"(kptr),
							"r"(biasptr),     // %20
							"r"(inch)         // %21
							: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25"
							);
					}
				}
#endif // __ARM_NEON && __aarch64__

				nn_outch = (outch - remain_outch_start) >> 2;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
				for (int pp = 0; pp < nn_outch; pp++)
				{
					int p = remain_outch_start + pp * 4;

					float* outptr0 = top_data + (p + 0) * top_cstep;
					float* outptr1 = top_data + (p + 1) * top_cstep;
					float* outptr2 = top_data + (p + 2) * top_cstep;
					float* outptr3 = top_data + (p + 3) * top_cstep;

					const float zeros[4] = { 0.f, 0.f, 0.f, 0.f };
					const float* biasptr = this->bias_term_ ? bias + p : zeros;

					int i = 0;

					for (; i + 7 < size; i += 8)
					{
						const float* tmpptr = tmp_data + (i / 8) * tmp_cstep;
#if __ARM_NEON && __aarch64__
						const float* kptr = kernel_data + (p / 8 + (p % 8) / 4) * kernel_cstep;
#else
						const float* kptr = kernel_data + (p / 4) * kernel_cstep;
#endif // __ARM_NEON && __aarch64__

#if __ARM_NEON
#if __aarch64__
						asm volatile(
							"ld1    {v0.4s}, [%12]          \n"
							"dup    v8.4s, v0.s[0]          \n"
							"dup    v9.4s, v0.s[0]          \n"
							"dup    v10.4s, v0.s[1]         \n"
							"dup    v11.4s, v0.s[1]         \n"
							"dup    v12.4s, v0.s[2]         \n"
							"dup    v13.4s, v0.s[2]         \n"
							"dup    v14.4s, v0.s[3]         \n"
							"dup    v15.4s, v0.s[3]         \n"

							// inch loop
							"lsr    w4, %w13, #2            \n"// w4 = nn = inch >> 2
							"cmp    w4, #0                  \n"
							"beq    1f                      \n"

							"0:                             \n"

							"prfm   pldl1keep, [%4, #512]   \n"
							"ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64     \n"

							"prfm   pldl1keep, [%5, #512]   \n"
							"ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64     \n"

							"fmla   v8.4s, v4.4s, v0.s[0]   \n"
							"fmla   v10.4s, v4.4s, v0.s[1]  \n"
							"fmla   v12.4s, v4.4s, v0.s[2]  \n"
							"fmla   v14.4s, v4.4s, v0.s[3]  \n"

							"fmla   v9.4s, v5.4s, v0.s[0]   \n"
							"fmla   v11.4s, v5.4s, v0.s[1]  \n"
							"fmla   v13.4s, v5.4s, v0.s[2]  \n"
							"fmla   v15.4s, v5.4s, v0.s[3]  \n"

							"prfm   pldl1keep, [%4, #512]   \n"
							"ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%4], #64 \n"

							"fmla   v8.4s, v6.4s, v1.s[0]   \n"
							"fmla   v10.4s, v6.4s, v1.s[1]  \n"
							"fmla   v12.4s, v6.4s, v1.s[2]  \n"
							"fmla   v14.4s, v6.4s, v1.s[3]  \n"

							"fmla   v9.4s, v7.4s, v1.s[0]   \n"
							"fmla   v11.4s, v7.4s, v1.s[1]  \n"
							"fmla   v13.4s, v7.4s, v1.s[2]  \n"
							"fmla   v15.4s, v7.4s, v1.s[3]  \n"

							"subs   w4, w4, #1              \n"

							"fmla   v8.4s, v16.4s, v2.s[0]  \n"
							"fmla   v10.4s, v16.4s, v2.s[1] \n"
							"fmla   v12.4s, v16.4s, v2.s[2] \n"
							"fmla   v14.4s, v16.4s, v2.s[3] \n"

							"fmla   v9.4s, v17.4s, v2.s[0]  \n"
							"fmla   v11.4s, v17.4s, v2.s[1] \n"
							"fmla   v13.4s, v17.4s, v2.s[2] \n"
							"fmla   v15.4s, v17.4s, v2.s[3] \n"

							"fmla   v8.4s, v18.4s, v3.s[0]  \n"
							"fmla   v10.4s, v18.4s, v3.s[1] \n"
							"fmla   v12.4s, v18.4s, v3.s[2] \n"
							"fmla   v14.4s, v18.4s, v3.s[3] \n"

							"fmla   v9.4s, v19.4s, v3.s[0]  \n"
							"fmla   v11.4s, v19.4s, v3.s[1] \n"
							"fmla   v13.4s, v19.4s, v3.s[2] \n"
							"fmla   v15.4s, v19.4s, v3.s[3] \n"

							"bne    0b                      \n"

							"1:                             \n"

							// remain loop
							"and    w4, %w13, #3            \n"// w4 = remain = inch & 3;
							"cmp    w4, #0                  \n"
							"beq    3f                      \n"

							"2:                             \n"

							"prfm   pldl1keep, [%4, #256]   \n"
							"ld1    {v4.4s, v5.4s}, [%4], #32   \n"

							"prfm   pldl1keep, [%5, #128]   \n"
							"ld1    {v0.4s}, [%5], #16      \n"

							"fmla   v8.4s, v4.4s, v0.s[0]   \n"
							"fmla   v10.4s, v4.4s, v0.s[1]  \n"
							"fmla   v12.4s, v4.4s, v0.s[2]  \n"
							"fmla   v14.4s, v4.4s, v0.s[3]  \n"

							"subs   w4, w4, #1              \n"

							"fmla   v9.4s, v5.4s, v0.s[0]   \n"
							"fmla   v11.4s, v5.4s, v0.s[1]  \n"
							"fmla   v13.4s, v5.4s, v0.s[2]  \n"
							"fmla   v15.4s, v5.4s, v0.s[3]  \n"

							"bne    2b                      \n"

							"3:                             \n"

							"st1    {v8.4s, v9.4s}, [%0], #32   \n"
							"st1    {v10.4s, v11.4s}, [%1], #32 \n"
							"st1    {v12.4s, v13.4s}, [%2], #32 \n"
							"st1    {v14.4s, v15.4s}, [%3], #32 \n"

							: "=r"(outptr0),    // %0
							"=r"(outptr1),    // %1
							"=r"(outptr2),    // %2
							"=r"(outptr3),    // %3
							"=r"(tmpptr),     // %4
							"=r"(kptr)        // %5
							: "0"(outptr0),
							"1"(outptr1),
							"2"(outptr2),
							"3"(outptr3),
							"4"(tmpptr),
							"5"(kptr),
							"r"(biasptr),     // %12
							"r"(inch)         // %13
							: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19"
							);
#else // __aarch64__
						asm volatile(
							"vld1.f32   {d0-d1}, [%12]      \n"
							"vdup.f32   q8, d0[0]           \n"
							"vdup.f32   q9, d0[0]           \n"
							"vdup.f32   q10, d0[1]          \n"
							"vdup.f32   q11, d0[1]          \n"
							"vdup.f32   q12, d1[0]          \n"
							"vdup.f32   q13, d1[0]          \n"
							"vdup.f32   q14, d1[1]          \n"
							"vdup.f32   q15, d1[1]          \n"

							// inch loop
							"lsr        r4, %13, #2         \n"// r4 = nn = inch >> 2
							"cmp        r4, #0              \n"
							"beq        1f                  \n"

							"0:                             \n"

							"pld        [%4, #512]          \n"
							"vldm       %4!, {d8-d15}       \n"
							//                 "vld1.f32   {d8-d11}, [%4 :128]!    \n"
							//                 "vld1.f32   {d12-d15}, [%4 :128]!   \n"

							"pld        [%5, #512]          \n"
							"vldm       %5!, {d0-d7}       \n"
							//                 "vld1.f32   {d0-d3}, [%5 :128]! \n"
							//                 "vld1.f32   {d4-d7}, [%5 :128]! \n"

							"vmla.f32   q8, q4, d0[0]       \n"
							"vmla.f32   q10, q4, d0[1]      \n"
							"vmla.f32   q12, q4, d1[0]      \n"
							"vmla.f32   q14, q4, d1[1]      \n"

							"vmla.f32   q9, q5, d0[0]       \n"
							"vmla.f32   q11, q5, d0[1]      \n"
							"vmla.f32   q13, q5, d1[0]      \n"
							"vmla.f32   q15, q5, d1[1]      \n"

							"vmla.f32   q8, q6, d2[0]       \n"
							"vmla.f32   q10, q6, d2[1]      \n"
							"vmla.f32   q12, q6, d3[0]      \n"
							"vmla.f32   q14, q6, d3[1]      \n"

							"vmla.f32   q9, q7, d2[0]       \n"
							"vmla.f32   q11, q7, d2[1]      \n"
							"vmla.f32   q13, q7, d3[0]      \n"
							"vmla.f32   q15, q7, d3[1]      \n"

							"pld        [%4, #512]          \n"
							"vldm       %4!, {d8-d15}       \n"
							//                 "vld1.f32   {d8-d11}, [%4 :128]!    \n"
							//                 "vld1.f32   {d12-d15}, [%4 :128]!   \n"

							"vmla.f32   q8, q4, d4[0]       \n"
							"vmla.f32   q10, q4, d4[1]      \n"
							"vmla.f32   q12, q4, d5[0]      \n"
							"vmla.f32   q14, q4, d5[1]      \n"

							"vmla.f32   q9, q5, d4[0]       \n"
							"vmla.f32   q11, q5, d4[1]      \n"
							"vmla.f32   q13, q5, d5[0]      \n"
							"vmla.f32   q15, q5, d5[1]      \n"

							"subs       r4, r4, #1          \n"

							"vmla.f32   q8, q6, d6[0]       \n"
							"vmla.f32   q10, q6, d6[1]      \n"
							"vmla.f32   q12, q6, d7[0]      \n"
							"vmla.f32   q14, q6, d7[1]      \n"

							"vmla.f32   q9, q7, d6[0]       \n"
							"vmla.f32   q11, q7, d6[1]      \n"
							"vmla.f32   q13, q7, d7[0]      \n"
							"vmla.f32   q15, q7, d7[1]      \n"

							"bne        0b                  \n"

							"1:                             \n"

							// remain loop
							"and        r4, %13, #3         \n"// r4 = remain = inch & 3;
							"cmp        r4, #0              \n"
							"beq        3f                  \n"

							"2:                             \n"

							"pld        [%4, #256]          \n"
							"vld1.f32   {d8-d11}, [%4 :128]!    \n"

							"pld        [%5, #128]          \n"
							"vld1.f32   {d0-d1}, [%5 :128]!     \n"

							"vmla.f32   q8, q4, d0[0]       \n"
							"vmla.f32   q10, q4, d0[1]      \n"
							"vmla.f32   q12, q4, d1[0]      \n"
							"vmla.f32   q14, q4, d1[1]      \n"

							"subs       r4, r4, #1          \n"

							"vmla.f32   q9, q5, d0[0]       \n"
							"vmla.f32   q11, q5, d0[1]      \n"
							"vmla.f32   q13, q5, d1[0]      \n"
							"vmla.f32   q15, q5, d1[1]      \n"

							"bne        2b                  \n"

							"3:                             \n"

							"vst1.f32   {d16-d19}, [%0 :128]!   \n"
							"vst1.f32   {d20-d23}, [%1 :128]!   \n"
							"vst1.f32   {d24-d27}, [%2 :128]!   \n"
							"vst1.f32   {d28-d31}, [%3 :128]!   \n"

							: "=r"(outptr0),    // %0
							"=r"(outptr1),    // %1
							"=r"(outptr2),    // %2
							"=r"(outptr3),    // %3
							"=r"(tmpptr),     // %4
							"=r"(kptr)        // %5
							: "0"(outptr0),
							"1"(outptr1),
							"2"(outptr2),
							"3"(outptr3),
							"4"(tmpptr),
							"5"(kptr),
							"r"(biasptr),     // %12
							"r"(inch)         // %13
							: "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
							);
#endif // __aarch64__
#else
						float sum0_0 = biasptr[0];
						float sum0_1 = biasptr[0];
						float sum0_2 = biasptr[0];
						float sum0_3 = biasptr[0];
						float sum0_4 = biasptr[0];
						float sum0_5 = biasptr[0];
						float sum0_6 = biasptr[0];
						float sum0_7 = biasptr[0];

						float sum1_0 = biasptr[1];
						float sum1_1 = biasptr[1];
						float sum1_2 = biasptr[1];
						float sum1_3 = biasptr[1];
						float sum1_4 = biasptr[1];
						float sum1_5 = biasptr[1];
						float sum1_6 = biasptr[1];
						float sum1_7 = biasptr[1];

						float sum2_0 = biasptr[2];
						float sum2_1 = biasptr[2];
						float sum2_2 = biasptr[2];
						float sum2_3 = biasptr[2];
						float sum2_4 = biasptr[2];
						float sum2_5 = biasptr[2];
						float sum2_6 = biasptr[2];
						float sum2_7 = biasptr[2];

						float sum3_0 = biasptr[3];
						float sum3_1 = biasptr[3];
						float sum3_2 = biasptr[3];
						float sum3_3 = biasptr[3];
						float sum3_4 = biasptr[3];
						float sum3_5 = biasptr[3];
						float sum3_6 = biasptr[3];
						float sum3_7 = biasptr[3];

						for (int q = 0; q < inch; q++)
						{
							sum0_0 += tmpptr[0] * kptr[0];
							sum0_1 += tmpptr[1] * kptr[0];
							sum0_2 += tmpptr[2] * kptr[0];
							sum0_3 += tmpptr[3] * kptr[0];
							sum0_4 += tmpptr[4] * kptr[0];
							sum0_5 += tmpptr[5] * kptr[0];
							sum0_6 += tmpptr[6] * kptr[0];
							sum0_7 += tmpptr[7] * kptr[0];

							sum1_0 += tmpptr[0] * kptr[1];
							sum1_1 += tmpptr[1] * kptr[1];
							sum1_2 += tmpptr[2] * kptr[1];
							sum1_3 += tmpptr[3] * kptr[1];
							sum1_4 += tmpptr[4] * kptr[1];
							sum1_5 += tmpptr[5] * kptr[1];
							sum1_6 += tmpptr[6] * kptr[1];
							sum1_7 += tmpptr[7] * kptr[1];

							sum2_0 += tmpptr[0] * kptr[2];
							sum2_1 += tmpptr[1] * kptr[2];
							sum2_2 += tmpptr[2] * kptr[2];
							sum2_3 += tmpptr[3] * kptr[2];
							sum2_4 += tmpptr[4] * kptr[2];
							sum2_5 += tmpptr[5] * kptr[2];
							sum2_6 += tmpptr[6] * kptr[2];
							sum2_7 += tmpptr[7] * kptr[2];

							sum3_0 += tmpptr[0] * kptr[3];
							sum3_1 += tmpptr[1] * kptr[3];
							sum3_2 += tmpptr[2] * kptr[3];
							sum3_3 += tmpptr[3] * kptr[3];
							sum3_4 += tmpptr[4] * kptr[3];
							sum3_5 += tmpptr[5] * kptr[3];
							sum3_6 += tmpptr[6] * kptr[3];
							sum3_7 += tmpptr[7] * kptr[3];

							tmpptr += 8;
							kptr += 4;
						}

						outptr0[0] = sum0_0;
						outptr0[1] = sum0_1;
						outptr0[2] = sum0_2;
						outptr0[3] = sum0_3;
						outptr0[4] = sum0_4;
						outptr0[5] = sum0_5;
						outptr0[6] = sum0_6;
						outptr0[7] = sum0_7;

						outptr1[0] = sum1_0;
						outptr1[1] = sum1_1;
						outptr1[2] = sum1_2;
						outptr1[3] = sum1_3;
						outptr1[4] = sum1_4;
						outptr1[5] = sum1_5;
						outptr1[6] = sum1_6;
						outptr1[7] = sum1_7;

						outptr2[0] = sum2_0;
						outptr2[1] = sum2_1;
						outptr2[2] = sum2_2;
						outptr2[3] = sum2_3;
						outptr2[4] = sum2_4;
						outptr2[5] = sum2_5;
						outptr2[6] = sum2_6;
						outptr2[7] = sum2_7;

						outptr3[0] = sum3_0;
						outptr3[1] = sum3_1;
						outptr3[2] = sum3_2;
						outptr3[3] = sum3_3;
						outptr3[4] = sum3_4;
						outptr3[5] = sum3_5;
						outptr3[6] = sum3_6;
						outptr3[7] = sum3_7;

						outptr0 += 8;
						outptr1 += 8;
						outptr2 += 8;
						outptr3 += 8;

#endif // __ARM_NEON
					}

					for (; i + 3 < size; i += 4)
					{
						const float* tmpptr = tmp_data + (i / 8 + (i % 8) / 4) * tmp_cstep;
#if __ARM_NEON && __aarch64__
						const float* kptr = kernel_data + (p / 8 + (p % 8) / 4) * kernel_cstep;
#else
						const float* kptr = kernel_data + (p / 4) * kernel_cstep;
#endif // __ARM_NEON && __aarch64__

#if __ARM_NEON
#if __aarch64__
						asm volatile(
							"ld1    {v0.4s}, [%12]          \n"
							"dup    v8.4s, v0.s[0]          \n"
							"dup    v9.4s, v0.s[1]          \n"
							"dup    v10.4s, v0.s[2]         \n"
							"dup    v11.4s, v0.s[3]         \n"

							// inch loop
							"lsr    w4, %w13, #2            \n"// w4 = nn = inch >> 2
							"cmp    w4, #0                  \n"
							"beq    1f                      \n"

							"0:                             \n"

							"prfm   pldl1keep, [%4, #512]   \n"
							"ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64     \n"

							"prfm   pldl1keep, [%5, #512]   \n"
							"ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64     \n"

							"fmla   v8.4s, v4.4s, v0.s[0]   \n"
							"fmla   v9.4s, v4.4s, v0.s[1]   \n"
							"fmla   v10.4s, v4.4s, v0.s[2]  \n"
							"fmla   v11.4s, v4.4s, v0.s[3]  \n"

							"fmla   v8.4s, v5.4s, v1.s[0]   \n"
							"fmla   v9.4s, v5.4s, v1.s[1]   \n"
							"fmla   v10.4s, v5.4s, v1.s[2]  \n"
							"fmla   v11.4s, v5.4s, v1.s[3]  \n"

							"subs   w4, w4, #1              \n"

							"fmla   v8.4s, v6.4s, v2.s[0]   \n"
							"fmla   v9.4s, v6.4s, v2.s[1]   \n"
							"fmla   v10.4s, v6.4s, v2.s[2]  \n"
							"fmla   v11.4s, v6.4s, v2.s[3]  \n"

							"fmla   v8.4s, v7.4s, v3.s[0]   \n"
							"fmla   v9.4s, v7.4s, v3.s[1]   \n"
							"fmla   v10.4s, v7.4s, v3.s[2]  \n"
							"fmla   v11.4s, v7.4s, v3.s[3]  \n"

							"bne    0b                      \n"

							"1:                             \n"

							// remain loop
							"and    w4, %w13, #3            \n"// w4 = remain = inch & 3;
							"cmp    w4, #0                  \n"
							"beq    3f                      \n"

							"2:                             \n"

							"prfm   pldl1keep, [%4, #128]   \n"
							"ld1    {v4.4s}, [%4], #16      \n"

							"prfm   pldl1keep, [%5, #128]   \n"
							"ld1    {v0.4s}, [%5], #16      \n"

							"subs   w4, w4, #1              \n"

							"fmla   v8.4s, v4.4s, v0.s[0]   \n"
							"fmla   v9.4s, v4.4s, v0.s[1]   \n"
							"fmla   v10.4s, v4.4s, v0.s[2]  \n"
							"fmla   v11.4s, v4.4s, v0.s[3]  \n"

							"bne    2b                      \n"

							"3:                             \n"

							"st1    {v8.4s}, [%0], #16      \n"
							"st1    {v9.4s}, [%1], #16      \n"
							"st1    {v10.4s}, [%2], #16     \n"
							"st1    {v11.4s}, [%3], #16     \n"

							: "=r"(outptr0),    // %0
							"=r"(outptr1),    // %1
							"=r"(outptr2),    // %2
							"=r"(outptr3),    // %3
							"=r"(tmpptr),     // %4
							"=r"(kptr)        // %5
							: "0"(outptr0),
							"1"(outptr1),
							"2"(outptr2),
							"3"(outptr3),
							"4"(tmpptr),
							"5"(kptr),
							"r"(biasptr),     // %12
							"r"(inch)         // %13
							: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"
							);
#else // __aarch64__
						asm volatile(
							"vld1.f32   {d0-d1}, [%12]      \n"
							"vdup.f32   q8, d0[0]           \n"
							"vdup.f32   q9, d0[1]           \n"
							"vdup.f32   q10, d1[0]          \n"
							"vdup.f32   q11, d1[1]          \n"

							// inch loop
							"lsr        r4, %13, #2         \n"// r4 = nn = inch >> 2
							"cmp        r4, #0              \n"
							"beq        1f                  \n"

							"0:                             \n"

							"pld        [%4, #512]          \n"
							"vldm       %4!, {d8-d15}       \n"
							//                 "vld1.f32   {d8-d11}, [%4 :128]!    \n"
							//                 "vld1.f32   {d12-d15}, [%4 :128]!   \n"

							"pld        [%5, #512]          \n"
							"vldm       %5!, {d0-d7}       \n"
							//                 "vld1.f32   {d0-d3}, [%5 :128]! \n"
							//                 "vld1.f32   {d4-d7}, [%5 :128]! \n"

							"vmla.f32   q8, q4, d0[0]       \n"
							"vmla.f32   q9, q4, d0[1]       \n"
							"vmla.f32   q10, q4, d1[0]      \n"
							"vmla.f32   q11, q4, d1[1]      \n"

							"vmla.f32   q8, q5, d2[0]       \n"
							"vmla.f32   q9, q5, d2[1]       \n"
							"vmla.f32   q10, q5, d3[0]      \n"
							"vmla.f32   q11, q5, d3[1]      \n"

							"subs       r4, r4, #1          \n"

							"vmla.f32   q8, q6, d4[0]       \n"
							"vmla.f32   q9, q6, d4[1]       \n"
							"vmla.f32   q10, q6, d5[0]      \n"
							"vmla.f32   q11, q6, d5[1]      \n"

							"vmla.f32   q8, q7, d6[0]       \n"
							"vmla.f32   q9, q7, d6[1]       \n"
							"vmla.f32   q10, q7, d7[0]      \n"
							"vmla.f32   q11, q7, d7[1]      \n"

							"bne        0b                  \n"

							"1:                             \n"

							// remain loop
							"and        r4, %13, #3         \n"// r4 = remain = inch & 3;
							"cmp        r4, #0              \n"
							"beq        3f                  \n"

							"2:                             \n"

							"pld        [%4, #128]          \n"
							"vld1.f32   {d8-d9}, [%4 :128]! \n"

							"pld        [%5, #128]          \n"
							"vld1.f32   {d0-d1}, [%5 :128]! \n"

							"subs       r4, r4, #1          \n"

							"vmla.f32   q8, q4, d0[0]       \n"
							"vmla.f32   q9, q4, d0[1]       \n"
							"vmla.f32   q10, q4, d1[0]      \n"
							"vmla.f32   q11, q4, d1[1]      \n"

							"bne        2b                  \n"

							"3:                             \n"

							"vst1.f32   {d16-d17}, [%0 :128]!   \n"
							"vst1.f32   {d18-d19}, [%1 :128]!   \n"
							"vst1.f32   {d20-d21}, [%2 :128]!   \n"
							"vst1.f32   {d22-d23}, [%3 :128]!   \n"

							: "=r"(outptr0),    // %0
							"=r"(outptr1),    // %1
							"=r"(outptr2),    // %2
							"=r"(outptr3),    // %3
							"=r"(tmpptr),     // %4
							"=r"(kptr)        // %5
							: "0"(outptr0),
							"1"(outptr1),
							"2"(outptr2),
							"3"(outptr3),
							"4"(tmpptr),
							"5"(kptr),
							"r"(biasptr),     // %12
							"r"(inch)         // %13
							: "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11"
							);
#endif // __aarch64__
#else
						float sum0_0 = biasptr[0];
						float sum0_1 = biasptr[0];
						float sum0_2 = biasptr[0];
						float sum0_3 = biasptr[0];

						float sum1_0 = biasptr[1];
						float sum1_1 = biasptr[1];
						float sum1_2 = biasptr[1];
						float sum1_3 = biasptr[1];

						float sum2_0 = biasptr[2];
						float sum2_1 = biasptr[2];
						float sum2_2 = biasptr[2];
						float sum2_3 = biasptr[2];

						float sum3_0 = biasptr[3];
						float sum3_1 = biasptr[3];
						float sum3_2 = biasptr[3];
						float sum3_3 = biasptr[3];

						for (int q = 0; q < inch; q++)
						{
							sum0_0 += tmpptr[0] * kptr[0];
							sum0_1 += tmpptr[1] * kptr[0];
							sum0_2 += tmpptr[2] * kptr[0];
							sum0_3 += tmpptr[3] * kptr[0];

							sum1_0 += tmpptr[0] * kptr[1];
							sum1_1 += tmpptr[1] * kptr[1];
							sum1_2 += tmpptr[2] * kptr[1];
							sum1_3 += tmpptr[3] * kptr[1];

							sum2_0 += tmpptr[0] * kptr[2];
							sum2_1 += tmpptr[1] * kptr[2];
							sum2_2 += tmpptr[2] * kptr[2];
							sum2_3 += tmpptr[3] * kptr[2];

							sum3_0 += tmpptr[0] * kptr[3];
							sum3_1 += tmpptr[1] * kptr[3];
							sum3_2 += tmpptr[2] * kptr[3];
							sum3_3 += tmpptr[3] * kptr[3];

							tmpptr += 4;
							kptr += 4;
						}

						outptr0[0] = sum0_0;
						outptr0[1] = sum0_1;
						outptr0[2] = sum0_2;
						outptr0[3] = sum0_3;

						outptr1[0] = sum1_0;
						outptr1[1] = sum1_1;
						outptr1[2] = sum1_2;
						outptr1[3] = sum1_3;

						outptr2[0] = sum2_0;
						outptr2[1] = sum2_1;
						outptr2[2] = sum2_2;
						outptr2[3] = sum2_3;

						outptr3[0] = sum3_0;
						outptr3[1] = sum3_1;
						outptr3[2] = sum3_2;
						outptr3[3] = sum3_3;

						outptr0 += 4;
						outptr1 += 4;
						outptr2 += 4;
						outptr3 += 4;
#endif // __ARM_NEON
					}

					for (; i < size; i++)
					{
						const float* tmpptr = tmp_data + (i / 8 + (i % 8) / 4 + i % 4) * tmp_cstep;
#if __ARM_NEON && __aarch64__
						const float* kptr = kernel_data + (p / 8 + (p % 8) / 4) * kernel_cstep;
#else
						const float* kptr = kernel_data + (p / 4) * kernel_cstep;
#endif // __ARM_NEON && __aarch64__

#if __ARM_NEON
#if __aarch64__
						asm volatile(
							"ld1    {v12.4s}, [%12]         \n"

							// inch loop
							"lsr    w4, %w13, #2            \n"// w4 = nn = inch >> 2
							"cmp    w4, #0                  \n"
							"beq    1f                      \n"

							"eor    v8.16b, v8.16b, v8.16b  \n"
							"eor    v9.16b, v9.16b, v9.16b  \n"
							"eor    v10.16b, v10.16b, v10.16b  \n"
							"eor    v11.16b, v11.16b, v11.16b  \n"

							"0:                             \n"

							"prfm   pldl1keep, [%4, #128]   \n"
							"ld1    {v4.4s}, [%4], #16      \n"

							"prfm   pldl1keep, [%5, #512]   \n"
							"ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64     \n"

							"subs   w4, w4, #1              \n"

							"fmla   v8.4s, v0.4s, v4.s[0]   \n"
							"fmla   v9.4s, v1.4s, v4.s[1]   \n"
							"fmla   v10.4s, v2.4s, v4.s[2]  \n"
							"fmla   v11.4s, v3.4s, v4.s[3]  \n"

							"bne    0b                      \n"

							"fadd   v8.4s, v8.4s, v9.4s     \n"
							"fadd   v10.4s, v10.4s, v11.4s  \n"
							"fadd   v8.4s, v8.4s, v10.4s    \n"
							"fadd   v12.4s, v12.4s, v8.4s   \n"

							"1:                             \n"

							// remain loop
							"and    w4, %w13, #3            \n"// w4 = remain = inch & 3;
							"cmp    w4, #0                  \n"
							"beq    3f                      \n"

							"2:                             \n"

							"prfm   pldl1keep, [%4, #32]    \n"
							"ld1r   {v4.4s}, [%4], #4       \n"

							"prfm   pldl1keep, [%5, #128]   \n"
							"ld1    {v0.4s}, [%5], #16      \n"

							"subs   w4, w4, #1              \n"

							"fmla   v12.4s, v4.4s, v0.4s    \n"

							"bne    2b                      \n"

							"3:                             \n"

							"st1    {v12.s}[0], [%0], #4    \n"
							"st1    {v12.s}[1], [%1], #4    \n"
							"st1    {v12.s}[2], [%2], #4    \n"
							"st1    {v12.s}[3], [%3], #4    \n"

							: "=r"(outptr0),    // %0
							"=r"(outptr1),    // %1
							"=r"(outptr2),    // %2
							"=r"(outptr3),    // %3
							"=r"(tmpptr),     // %4
							"=r"(kptr)        // %5
							: "0"(outptr0),
							"1"(outptr1),
							"2"(outptr2),
							"3"(outptr3),
							"4"(tmpptr),
							"5"(kptr),
							"r"(biasptr),     // %12
							"r"(inch)         // %13
							: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10", "v11", "v12"
							);
#else // __aarch64__
						asm volatile(
							"vld1.f32   {d24-d25}, [%12]    \n"

							// inch loop
							"lsr        r4, %13, #2         \n"// r4 = nn = inch >> 2
							"cmp        r4, #0              \n"
							"beq        1f                  \n"

							"veor       q8, q8, q8          \n"
							"veor       q9, q9, q9          \n"
							"veor       q10, q10, q10       \n"
							"veor       q11, q11, q11       \n"

							"0:                             \n"

							"pld        [%4, #128]          \n"
							"vld1.f32   {d8-d9}, [%4 :128]! \n"

							"pld        [%5, #512]          \n"
							"vldm       %5!, {d0-d7}       \n"
							//                 "vld1.f32   {d0-d3}, [%5 :128]! \n"
							//                 "vld1.f32   {d4-d7}, [%5 :128]! \n"

							"subs       r4, r4, #1          \n"

							"vmla.f32   q8, q0, d8[0]       \n"
							"vmla.f32   q9, q1, d8[1]       \n"
							"vmla.f32   q10, q2, d9[0]      \n"
							"vmla.f32   q11, q3, d9[1]      \n"

							"bne        0b                  \n"

							"vadd.f32   q8, q8, q9          \n"
							"vadd.f32   q10, q10, q11       \n"
							"vadd.f32   q8, q8, q10         \n"
							"vadd.f32   q12, q12, q8        \n"

							"1:                             \n"

							// remain loop
							"and        r4, %13, #3         \n"// r4 = remain = inch & 3;
							"cmp        r4, #0              \n"
							"beq        3f                  \n"

							"2:                             \n"

							"pld        [%4, #32]           \n"
							"vld1.f32   {d8[],d9[]}, [%4]!  \n"

							"pld        [%5, #128]          \n"
							"vld1.f32   {d0-d1}, [%5 :128]! \n"

							"subs       r4, r4, #1          \n"

							"vmla.f32   q12, q4, q0         \n"

							"bne        2b                  \n"

							"3:                             \n"

							"vst1.f32   {d24[0]}, [%0]!     \n"
							"vst1.f32   {d24[1]}, [%1]!     \n"
							"vst1.f32   {d25[0]}, [%2]!     \n"
							"vst1.f32   {d25[1]}, [%3]!     \n"

							: "=r"(outptr0),    // %0
							"=r"(outptr1),    // %1
							"=r"(outptr2),    // %2
							"=r"(outptr3),    // %3
							"=r"(tmpptr),     // %4
							"=r"(kptr)        // %5
							: "0"(outptr0),
							"1"(outptr1),
							"2"(outptr2),
							"3"(outptr3),
							"4"(tmpptr),
							"5"(kptr),
							"r"(biasptr),     // %12
							"r"(inch)         // %13
							: "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q8", "q9", "q10", "q11", "q12"
							);
#endif // __aarch64__
#else
						float sum0 = biasptr[0];
						float sum1 = biasptr[1];
						float sum2 = biasptr[2];
						float sum3 = biasptr[3];

						for (int q = 0; q < inch; q++)
						{
							sum0 += tmpptr[0] * kptr[0];
							sum1 += tmpptr[0] * kptr[1];
							sum2 += tmpptr[0] * kptr[2];
							sum3 += tmpptr[0] * kptr[3];

							tmpptr++;
							kptr += 4;
						}

						outptr0[0] = sum0;
						outptr1[0] = sum1;
						outptr2[0] = sum2;
						outptr3[0] = sum3;

						outptr0++;
						outptr1++;
						outptr2++;
						outptr3++;
#endif // __ARM_NEON
					}
				}

				remain_outch_start += nn_outch << 2;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
				for (int p = remain_outch_start; p < outch; p++)
				{
					float* outptr0 = top_data + (p)* top_cstep;

					const float bias0 = this->bias_term_ ? bias[p] : 0.f;

					int i = 0;

					for (; i + 7 < size; i += 8)
					{
						const float* tmpptr = tmp_data + (i / 8) * tmp_cstep;
#if __ARM_NEON && __aarch64__
						const float* kptr = kernel_data + (p / 8 + (p % 8) / 4 + p % 4) * kernel_cstep;
#else
						const float* kptr = kernel_data + (p / 4 + p % 4) * kernel_cstep;
#endif // __ARM_NEON && __aarch64__

#if __ARM_NEON
#if __aarch64__
						asm volatile(
							"dup    v8.4s, %w6              \n"
							"dup    v9.4s, %w6              \n"

							// inch loop
							"lsr    w4, %w7, #2             \n"// w4 = nn = inch >> 2
							"cmp    w4, #0                  \n"
							"beq    1f                      \n"

							"0:                             \n"

							"prfm   pldl1keep, [%1, #512]   \n"
							"ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64     \n"

							"prfm   pldl1keep, [%2, #128]   \n"
							"ld1    {v0.4s}, [%2], #16      \n"

							"fmla   v8.4s, v4.4s, v0.s[0]   \n"
							"fmla   v9.4s, v5.4s, v0.s[0]   \n"

							"prfm   pldl1keep, [%1, #512]   \n"
							"ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%1], #64 \n"

							"fmla   v8.4s, v6.4s, v0.s[1]   \n"
							"fmla   v9.4s, v7.4s, v0.s[1]   \n"

							"subs   w4, w4, #1              \n"

							"fmla   v8.4s, v12.4s, v0.s[2]  \n"
							"fmla   v9.4s, v13.4s, v0.s[2]  \n"

							"fmla   v8.4s, v14.4s, v0.s[3]  \n"
							"fmla   v9.4s, v15.4s, v0.s[3]  \n"

							"bne    0b                      \n"

							"1:                             \n"

							// remain loop
							"and    w4, %w7, #3             \n"// w4 = remain = inch & 3;
							"cmp    w4, #0                  \n"
							"beq    3f                      \n"

							"2:                             \n"

							"prfm   pldl1keep, [%1, #256]   \n"
							"ld1    {v4.4s, v5.4s}, [%1], #32   \n"

							"prfm   pldl1keep, [%2, #32]    \n"
							"ld1r   {v0.4s}, [%2], #4       \n"

							"subs   w4, w4, #1              \n"

							"fmla   v8.4s, v4.4s, v0.4s     \n"
							"fmla   v9.4s, v5.4s, v0.4s     \n"

							"bne    2b                      \n"

							"3:                             \n"

							"st1    {v8.4s, v9.4s}, [%0], #32   \n"

							: "=r"(outptr0),    // %0
							"=r"(tmpptr),     // %1
							"=r"(kptr)        // %2
							: "0"(outptr0),
							"1"(tmpptr),
							"2"(kptr),
							"r"(bias0),       // %6
							"r"(inch)         // %7
							: "cc", "memory", "x4", "v0", "v4", "v5", "v6", "v7", "v8", "v9", "v12", "v13", "v14", "v15"
							);
#else // __aarch64__
						asm volatile(
							"vdup.f32   q8, %6              \n"
							"vdup.f32   q9, %6              \n"

							// inch loop
							"lsr        r4, %7, #2          \n"// r4 = nn = inch >> 2
							"cmp        r4, #0              \n"
							"beq        1f                  \n"

							"0:                             \n"

							"pld        [%1, #512]          \n"
							"vldm       %1!, {d8-d15}       \n"
							//                 "vld1.f32   {d8-d11}, [%1 :128]!    \n"
							//                 "vld1.f32   {d12-d15}, [%1 :128]!   \n"

							"pld        [%2, #128]          \n"
							"vld1.f32   {d0-d1}, [%2 :128]! \n"

							"vmla.f32   q8, q4, d0[0]       \n"
							"vmla.f32   q9, q5, d0[0]       \n"

							"pld        [%1, #512]          \n"
							"vldm       %1!, {d24-d31}      \n"
							//                 "vld1.f32   {d24-d27}, [%1 :128]!   \n"
							//                 "vld1.f32   {d28-d31}, [%1 :128]!   \n"

							"vmla.f32   q8, q6, d0[1]       \n"
							"vmla.f32   q9, q7, d0[1]       \n"

							"subs       r4, r4, #1          \n"

							"vmla.f32   q8, q12, d1[0]      \n"
							"vmla.f32   q9, q13, d1[0]      \n"

							"vmla.f32   q8, q14, d1[1]      \n"
							"vmla.f32   q9, q15, d1[1]      \n"

							"bne        0b                  \n"

							"1:                             \n"

							// remain loop
							"and        r4, %7, #3          \n"// r4 = remain = inch & 3;
							"cmp        r4, #0              \n"
							"beq        3f                  \n"

							"2:                             \n"

							"pld        [%1, #256]          \n"
							"vld1.f32   {d8-d11}, [%1 :128]!    \n"

							"pld        [%2, #32]           \n"
							"vld1.f32   {d0[],d1[]}, [%2]!  \n"

							"subs       r4, r4, #1          \n"

							"vmla.f32   q8, q4, q0          \n"
							"vmla.f32   q9, q5, q0          \n"

							"bne        2b                  \n"

							"3:                             \n"

							"vst1.f32   {d16-d19}, [%0 :128]!   \n"

							: "=r"(outptr0),    // %0
							"=r"(tmpptr),     // %1
							"=r"(kptr)        // %2
							: "0"(outptr0),
							"1"(tmpptr),
							"2"(kptr),
							"r"(bias0),       // %6
							"r"(inch)         // %7
							: "cc", "memory", "r4", "q0", "q4", "q5", "q6", "q7", "q8", "q9", "q12", "q13", "q14", "q15"
							);
#endif // __aarch64__
#else
						float sum0 = bias0;
						float sum1 = bias0;
						float sum2 = bias0;
						float sum3 = bias0;
						float sum4 = bias0;
						float sum5 = bias0;
						float sum6 = bias0;
						float sum7 = bias0;

						for (int q = 0; q < inch; q++)
						{
							sum0 += tmpptr[0] * kptr[0];
							sum1 += tmpptr[1] * kptr[0];
							sum2 += tmpptr[2] * kptr[0];
							sum3 += tmpptr[3] * kptr[0];
							sum4 += tmpptr[4] * kptr[0];
							sum5 += tmpptr[5] * kptr[0];
							sum6 += tmpptr[6] * kptr[0];
							sum7 += tmpptr[7] * kptr[0];

							tmpptr += 8;
							kptr++;
						}

						outptr0[0] = sum0;
						outptr0[1] = sum1;
						outptr0[2] = sum2;
						outptr0[3] = sum3;
						outptr0[4] = sum4;
						outptr0[5] = sum5;
						outptr0[6] = sum6;
						outptr0[7] = sum7;

						outptr0 += 8;
#endif // __ARM_NEON
					}

					for (; i + 3 < size; i += 4)
					{
						const float* tmpptr = tmp_data + (i / 8 + (i % 8) / 4) * tmp_cstep;
#if __ARM_NEON && __aarch64__
						const float* kptr = kernel_data + (p / 8 + (p % 8) / 4 + p % 4) * kernel_cstep;
#else
						const float* kptr = kernel_data + (p / 4 + p % 4) * kernel_cstep;
#endif // __ARM_NEON && __aarch64__

#if __ARM_NEON
#if __aarch64__
						asm volatile(
							"dup    v8.4s, %w6              \n"

							// inch loop
							"lsr    w4, %w7, #2             \n"// w4 = nn = inch >> 2
							"cmp    w4, #0                  \n"
							"beq    1f                      \n"

							"0:                             \n"

							"prfm   pldl1keep, [%1, #512]   \n"
							"ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64     \n"

							"prfm   pldl1keep, [%2, #128]   \n"
							"ld1    {v0.4s}, [%2], #16      \n"

							"subs   w4, w4, #1              \n"

							"fmla   v8.4s, v4.4s, v0.s[0]   \n"
							"fmla   v8.4s, v5.4s, v0.s[1]   \n"
							"fmla   v8.4s, v6.4s, v0.s[2]   \n"
							"fmla   v8.4s, v7.4s, v0.s[3]   \n"

							"bne    0b                      \n"

							"1:                             \n"

							// remain loop
							"and    w4, %w7, #3             \n"// w4 = remain = inch & 3;
							"cmp    w4, #0                  \n"
							"beq    3f                      \n"

							"2:                             \n"

							"prfm   pldl1keep, [%1, #128]   \n"
							"ld1    {v4.4s}, [%1], #16      \n"

							"prfm   pldl1keep, [%2, #32]    \n"
							"ld1r   {v0.4s}, [%2], #4       \n"

							"subs   w4, w4, #1              \n"

							"fmla   v8.4s, v4.4s, v0.4s     \n"

							"bne    2b                      \n"

							"3:                             \n"

							"st1    {v8.4s}, [%0], #16      \n"

							: "=r"(outptr0),    // %0
							"=r"(tmpptr),     // %1
							"=r"(kptr)        // %2
							: "0"(outptr0),
							"1"(tmpptr),
							"2"(kptr),
							"r"(bias0),       // %6
							"r"(inch)         // %7
							: "cc", "memory", "x4", "v0", "v4", "v5", "v6", "v7", "v8"
							);
#else // __aarch64__
						asm volatile(
							"vdup.f32   q8, %6              \n"

							// inch loop
							"lsr        r4, %7, #2          \n"// r4 = nn = inch >> 2
							"cmp        r4, #0              \n"
							"beq        1f                  \n"

							"0:                             \n"

							"pld        [%1, #512]          \n"
							"vldm       %1!, {d8-d15}       \n"
							//                 "vld1.f32   {d8-d11}, [%1 :128]!    \n"
							//                 "vld1.f32   {d12-d15}, [%1 :128]!   \n"

							"pld        [%2, #128]          \n"
							"vld1.f32   {d0-d1}, [%2]!      \n"

							"subs       r4, r4, #1          \n"

							"vmla.f32   q8, q4, d0[0]       \n"
							"vmla.f32   q8, q5, d0[1]       \n"
							"vmla.f32   q8, q6, d1[0]       \n"
							"vmla.f32   q8, q7, d1[1]       \n"

							"bne        0b                  \n"

							"1:                             \n"

							// remain loop
							"and        r4, %7, #3          \n"// r4 = remain = inch & 3;
							"cmp        r4, #0              \n"
							"beq        3f                  \n"

							"2:                             \n"

							"pld        [%1, #128]          \n"
							"vld1.f32   {d8-d9}, [%1 :128]! \n"

							"pld        [%2, #32]           \n"
							"vld1.f32   {d0[],d1[]}, [%2]!  \n"

							"subs       r4, r4, #1          \n"

							"vmla.f32   q8, q4, q0          \n"

							"bne        2b                  \n"

							"3:                             \n"

							"vst1.f32   {d16-d17}, [%0 :128]!   \n"

							: "=r"(outptr0),    // %0
							"=r"(tmpptr),     // %1
							"=r"(kptr)        // %2
							: "0"(outptr0),
							"1"(tmpptr),
							"2"(kptr),
							"r"(bias0),       // %6
							"r"(inch)         // %7
							: "cc", "memory", "r4", "q0", "q4", "q5", "q6", "q7", "q8"
							);
#endif // __aarch64__
#else
						float sum0 = bias0;
						float sum1 = bias0;
						float sum2 = bias0;
						float sum3 = bias0;

						for (int q = 0; q < inch; q++)
						{
							sum0 += tmpptr[0] * kptr[0];
							sum1 += tmpptr[1] * kptr[0];
							sum2 += tmpptr[2] * kptr[0];
							sum3 += tmpptr[3] * kptr[0];

							tmpptr += 4;
							kptr++;
						}

						outptr0[0] = sum0;
						outptr0[1] = sum1;
						outptr0[2] = sum2;
						outptr0[3] = sum3;

						outptr0 += 4;
#endif // __ARM_NEON
					}

					for (; i < size; i++)
					{
						const float* tmpptr = tmp_data + (i / 8 + (i % 8) / 4 + i % 4) * tmp_cstep;
#if __ARM_NEON && __aarch64__
						const float* kptr = kernel_data + (p / 8 + (p % 8) / 4 + p % 4) * tmp_cstep;
#else
						const float* kptr = kernel_data + (p / 4 + p % 4) * tmp_cstep;
#endif // __ARM_NEON && __aarch64__

						int q = 0;

#if __ARM_NEON
						float32x4_t _sum0 = vdupq_n_f32(0.f);

						for (; q + 3 < inch; q += 4)
						{
							float32x4_t _p0 = vld1q_f32(tmpptr);
							tmpptr += 4;

							float32x4_t _k0 = vld1q_f32(kptr);
							kptr += 4;

#if __aarch64__
							_sum0 = vfmaq_f32(_sum0, _p0, _k0);
#else
							_sum0 = vmlaq_f32(_sum0, _p0, _k0);
#endif
						}

#if __aarch64__
						float sum0 = bias0 + vaddvq_f32(_sum0);
#else
						float32x2_t _ss = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
						float sum0 = bias0 + vget_lane_f32(vpadd_f32(_ss, _ss), 0);
#endif
#else
						float sum0 = bias0;
#endif // __ARM_NEON

						for (; q < inch; q++)
						{
							sum0 += tmpptr[0] * kptr[0];
							tmpptr++;
							kptr++;
						}

						outptr0[0] = sum0;

						outptr0++;
					}
				}

				//     // NOTE sgemm
				//     for (; p<outch; p++)
				//     {
				//         Mat out0 = top_blob.channel(p);
				//
				//         const float bias0 = bias ? bias[p] : 0.f;
				//
				//         float* outptr0 = out0;
				//
				//         for (int i=0; i<size; i++)
				//         {
				//             float sum = bias0;
				//
				//             const float* kptr = _kernel.channel(p/8 + p%8);
				//
				//             for (int q=0; q<inch; q++)
				//             {
				//                 const float* img0 = bottom.channel(q);
				//
				//                 sum += img0[i] * kptr[0];
				//                 kptr ++;
				//             }
				//
				//             outptr0[i] = sum;
				//         }
				//     }
			}
		}

		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv1x1s1_neon(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top)
		{
			int num = bottom->num();
			int inch = bottom->channels();
			int bottom_cstep = bottom->width() * bottom->height();

			int outw = top->width();
			int outh = top->height();
			int outch = top->channels();
			int top_cstep = outw * outh;

			const float* kernel = this->weights_f32_[0]->cpu_data();

			const float* bias = nullptr;
			if (this->bias_term_)
				bias = this->weights_f32_[1]->cpu_data();

			for (int num_i = 0; num_i < num; num_i++)
			{
				const float *bottom_data = bottom->cpu_data() + num_i * inch * bottom_cstep;
				float *top_data = top->mutable_cpu_data() + num_i * outch * top_cstep;

				int nn_outch = 0;
				int remain_outch_start = 0;

#if __ARM_NEON && __aarch64__

				nn_outch = outch >> 3;
				remain_outch_start = nn_outch << 3;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
				for (int pp = 0; pp < nn_outch; pp++)
				{
					int p = pp * 8;

					float *out0 = top_data + (p + 0) * top_cstep;
					float *out1 = top_data + (p + 1) * top_cstep;
					float *out2 = top_data + (p + 2) * top_cstep;
					float *out3 = top_data + (p + 3) * top_cstep;
					float *out4 = top_data + (p + 4) * top_cstep;
					float *out5 = top_data + (p + 5) * top_cstep;
					float *out6 = top_data + (p + 6) * top_cstep;
					float *out7 = top_data + (p + 7) * top_cstep;

					const float bias0 = this->bias_term_ ? bias[p] : 0.f;
					const float bias1 = this->bias_term_ ? bias[p + 1] : 0.f;
					const float bias2 = this->bias_term_ ? bias[p + 2] : 0.f;
					const float bias3 = this->bias_term_ ? bias[p + 3] : 0.f;
					const float bias4 = this->bias_term_ ? bias[p + 4] : 0.f;
					const float bias5 = this->bias_term_ ? bias[p + 5] : 0.f;
					const float bias6 = this->bias_term_ ? bias[p + 6] : 0.f;
					const float bias7 = this->bias_term_ ? bias[p + 7] : 0.f;

					fill(out0, top_cstep, bias0);
					fill(out1, top_cstep, bias1);
					fill(out2, top_cstep, bias2);
					fill(out3, top_cstep, bias3);
					fill(out4, top_cstep, bias4);
					fill(out5, top_cstep, bias5);
					fill(out6, top_cstep, bias6);
					fill(out7, top_cstep, bias7);

					int q = 0;

					for (; q + 7 < inch; q += 8)
					{
						float* outptr0 = out0;
						float* outptr1 = out1;
						float* outptr2 = out2;
						float* outptr3 = out3;
						float* outptr4 = out4;
						float* outptr5 = out5;
						float* outptr6 = out6;
						float* outptr7 = out7;

						const float* img0 = bottom_data + (q + 0) * bottom_cstep;
						const float* img1 = bottom_data + (q + 1) * bottom_cstep;
						const float* img2 = bottom_data + (q + 2) * bottom_cstep;
						const float* img3 = bottom_data + (q + 3) * bottom_cstep;
						const float* img4 = bottom_data + (q + 4) * bottom_cstep;
						const float* img5 = bottom_data + (q + 5) * bottom_cstep;
						const float* img6 = bottom_data + (q + 6) * bottom_cstep;
						const float* img7 = bottom_data + (q + 7) * bottom_cstep;

						const float* kernel0 = kernel + p * inch + q;
						const float* kernel1 = kernel + (p + 1)*inch + q;
						const float* kernel2 = kernel + (p + 2)*inch + q;
						const float* kernel3 = kernel + (p + 3)*inch + q;
						const float* kernel4 = kernel + (p + 4)*inch + q;
						const float* kernel5 = kernel + (p + 5)*inch + q;
						const float* kernel6 = kernel + (p + 6)*inch + q;
						const float* kernel7 = kernel + (p + 7)*inch + q;

						const float* r0 = img0;
						const float* r1 = img1;
						const float* r2 = img2;
						const float* r3 = img3;
						const float* r4 = img4;
						const float* r5 = img5;
						const float* r6 = img6;
						const float* r7 = img7;

						int size = outw * outh;

						int nn = size >> 2;
						int remain = size & 3;

						float32x4_t _k0 = vld1q_f32(kernel0);
						float32x4_t _k1 = vld1q_f32(kernel1);
						float32x4_t _k2 = vld1q_f32(kernel2);
						float32x4_t _k3 = vld1q_f32(kernel3);
						float32x4_t _k4 = vld1q_f32(kernel4);
						float32x4_t _k5 = vld1q_f32(kernel5);
						float32x4_t _k6 = vld1q_f32(kernel6);
						float32x4_t _k7 = vld1q_f32(kernel7);

						float32x4_t _k0n = vld1q_f32(kernel0 + 4);
						float32x4_t _k1n = vld1q_f32(kernel1 + 4);
						float32x4_t _k2n = vld1q_f32(kernel2 + 4);
						float32x4_t _k3n = vld1q_f32(kernel3 + 4);
						float32x4_t _k4n = vld1q_f32(kernel4 + 4);
						float32x4_t _k5n = vld1q_f32(kernel5 + 4);
						float32x4_t _k6n = vld1q_f32(kernel6 + 4);
						float32x4_t _k7n = vld1q_f32(kernel7 + 4);

#ifdef __clang__
						// gcc reject over 30 oprands :(
						if (nn > 0)
						{
							asm volatile(
								"prfm   pldl1keep, [%9, #128]       \n"
								"ld1    {v17.4s}, [%9], #16         \n"

								"prfm   pldl1keep, [%1, #128]       \n"
								"ld1    {v18.4s}, [%1]              \n"

								"prfm   pldl1keep, [%2, #128]       \n"
								"ld1    {v19.4s}, [%2]              \n"

								"0:                                 \n"

								"fmla   v18.4s, v17.4s, %34.s[0]    \n"

								"prfm   pldl1keep, [%3, #128]       \n"
								"ld1    {v20.4s}, [%3]              \n"

								"fmla   v19.4s, v17.4s, %35.s[0]    \n"

								"prfm   pldl1keep, [%4, #128]       \n"
								"ld1    {v21.4s}, [%4]              \n"

								"fmla   v20.4s, v17.4s, %36.s[0]    \n"

								"prfm   pldl1keep, [%5, #128]       \n"
								"ld1    {v22.4s}, [%5]              \n"

								"fmla   v21.4s, v17.4s, %37.s[0]    \n"

								"prfm   pldl1keep, [%6, #128]       \n"
								"ld1    {v23.4s}, [%6]              \n"

								"fmla   v22.4s, v17.4s, %38.s[0]    \n"

								"prfm   pldl1keep, [%10, #128]      \n"
								"ld1    {v16.4s}, [%10], #16        \n"

								"fmla   v23.4s, v17.4s, %39.s[0]    \n"

								"prfm   pldl1keep, [%7, #128]       \n"
								"ld1    {v24.4s}, [%7]              \n"

								"fmla   v18.4s, v16.4s, %34.s[1]    \n"
								"fmla   v19.4s, v16.4s, %35.s[1]    \n"

								"prfm   pldl1keep, [%8, #128]       \n"
								"ld1    {v25.4s}, [%8]              \n"

								"fmla   v24.4s, v17.4s, %40.s[0]    \n"
								"fmla   v25.4s, v17.4s, %41.s[0]    \n"

								"fmla   v20.4s, v16.4s, %36.s[1]    \n"
								"fmla   v21.4s, v16.4s, %37.s[1]    \n"

								"prfm   pldl1keep, [%11, #128]      \n"
								"ld1    {v17.4s}, [%11], #16        \n"

								"fmla   v22.4s, v16.4s, %38.s[1]    \n"
								"fmla   v23.4s, v16.4s, %39.s[1]    \n"

								"fmla   v18.4s, v17.4s, %34.s[2]    \n"
								"fmla   v19.4s, v17.4s, %35.s[2]    \n"

								"fmla   v24.4s, v16.4s, %40.s[1]    \n"
								"fmla   v25.4s, v16.4s, %41.s[1]    \n"

								"fmla   v20.4s, v17.4s, %36.s[2]    \n"
								"fmla   v21.4s, v17.4s, %37.s[2]    \n"

								"prfm   pldl1keep, [%12, #128]      \n"
								"ld1    {v16.4s}, [%12], #16        \n"

								"fmla   v22.4s, v17.4s, %38.s[2]    \n"
								"fmla   v23.4s, v17.4s, %39.s[2]    \n"

								"fmla   v18.4s, v16.4s, %34.s[3]    \n"
								"fmla   v19.4s, v16.4s, %35.s[3]    \n"

								"fmla   v24.4s, v17.4s, %40.s[2]    \n"
								"fmla   v25.4s, v17.4s, %41.s[2]    \n"

								"fmla   v20.4s, v16.4s, %36.s[3]    \n"
								"fmla   v21.4s, v16.4s, %37.s[3]    \n"

								"prfm   pldl1keep, [%13, #128]      \n"
								"ld1    {v17.4s}, [%13], #16        \n"

								"fmla   v22.4s, v16.4s, %38.s[3]    \n"
								"fmla   v23.4s, v16.4s, %39.s[3]    \n"

								"fmla   v18.4s, v17.4s, %42.s[0]    \n"
								"fmla   v19.4s, v17.4s, %43.s[0]    \n"

								"fmla   v24.4s, v16.4s, %40.s[3]    \n"
								"fmla   v25.4s, v16.4s, %41.s[3]    \n"

								"fmla   v20.4s, v17.4s, %44.s[0]    \n"
								"fmla   v21.4s, v17.4s, %45.s[0]    \n"

								"prfm   pldl1keep, [%14, #128]      \n"
								"ld1    {v16.4s}, [%14], #16        \n"

								"fmla   v22.4s, v17.4s, %46.s[0]    \n"
								"fmla   v23.4s, v17.4s, %47.s[0]    \n"

								"fmla   v18.4s, v16.4s, %42.s[1]    \n"
								"fmla   v19.4s, v16.4s, %43.s[1]    \n"

								"fmla   v24.4s, v17.4s, %48.s[0]    \n"
								"fmla   v25.4s, v17.4s, %49.s[0]    \n"

								"fmla   v20.4s, v16.4s, %44.s[1]    \n"
								"fmla   v21.4s, v16.4s, %45.s[1]    \n"

								"prfm   pldl1keep, [%15, #128]      \n"
								"ld1    {v17.4s}, [%15], #16        \n"

								"fmla   v22.4s, v16.4s, %46.s[1]    \n"
								"fmla   v23.4s, v16.4s, %47.s[1]    \n"

								"fmla   v18.4s, v17.4s, %42.s[2]    \n"
								"fmla   v19.4s, v17.4s, %43.s[2]    \n"

								"fmla   v24.4s, v16.4s, %48.s[1]    \n"
								"fmla   v25.4s, v16.4s, %49.s[1]    \n"

								"fmla   v20.4s, v17.4s, %44.s[2]    \n"
								"fmla   v21.4s, v17.4s, %45.s[2]    \n"

								"prfm   pldl1keep, [%16, #128]      \n"
								"ld1    {v16.4s}, [%16], #16        \n"

								"fmla   v22.4s, v17.4s, %46.s[2]    \n"
								"fmla   v23.4s, v17.4s, %47.s[2]    \n"

								"fmla   v18.4s, v16.4s, %42.s[3]    \n"
								"fmla   v19.4s, v16.4s, %43.s[3]    \n"

								"fmla   v24.4s, v17.4s, %48.s[2]    \n"
								"fmla   v25.4s, v17.4s, %49.s[2]    \n"

								"fmla   v20.4s, v16.4s, %44.s[3]    \n"
								"fmla   v21.4s, v16.4s, %45.s[3]    \n"

								"st1    {v18.4s}, [%1], #16         \n"

								"fmla   v22.4s, v16.4s, %46.s[3]    \n"

								"st1    {v19.4s}, [%2], #16         \n"

								"fmla   v23.4s, v16.4s, %47.s[3]    \n"

								"st1    {v20.4s}, [%3], #16         \n"

								"prfm   pldl1keep, [%9, #128]       \n"
								"ld1    {v17.4s}, [%9], #16         \n"

								"fmla   v24.4s, v16.4s, %48.s[3]    \n"

								"st1    {v21.4s}, [%4], #16         \n"

								"fmla   v25.4s, v16.4s, %49.s[3]    \n"

								"st1    {v22.4s}, [%5], #16         \n"

								"prfm   pldl1keep, [%1, #128]       \n"
								"ld1    {v18.4s}, [%1]              \n"

								"st1    {v23.4s}, [%6], #16         \n"

								"prfm   pldl1keep, [%2, #128]       \n"
								"ld1    {v19.4s}, [%2]              \n"

								"st1    {v24.4s}, [%7], #16         \n"

								"subs   %w0, %w0, #1                \n"

								"st1    {v25.4s}, [%8], #16         \n"

								"bne    0b                          \n"
								"sub    %9, %9, #16                 \n"
								: "=r"(nn),     // %0
								"=r"(outptr0),// %1
								"=r"(outptr1),// %2
								"=r"(outptr2),// %3
								"=r"(outptr3),// %4
								"=r"(outptr4),// %5
								"=r"(outptr5),// %6
								"=r"(outptr6),// %7
								"=r"(outptr7),// %8
								"=r"(r0),     // %9
								"=r"(r1),     // %10
								"=r"(r2),     // %11
								"=r"(r3),     // %12
								"=r"(r4),     // %13
								"=r"(r5),     // %14
								"=r"(r6),     // %15
								"=r"(r7)      // %16
								: "0"(nn),
								"1"(outptr0),
								"2"(outptr1),
								"3"(outptr2),
								"4"(outptr3),
								"5"(outptr4),
								"6"(outptr5),
								"7"(outptr6),
								"8"(outptr7),
								"9"(r0),
								"10"(r1),
								"11"(r2),
								"12"(r3),
								"13"(r4),
								"14"(r5),
								"15"(r6),
								"16"(r7),
								"w"(_k0),     // %34
								"w"(_k1),     // %35
								"w"(_k2),     // %36
								"w"(_k3),     // %37
								"w"(_k4),     // %38
								"w"(_k5),     // %39
								"w"(_k6),     // %40
								"w"(_k7),     // %41
								"w"(_k0n),    // %42
								"w"(_k1n),    // %43
								"w"(_k2n),    // %44
								"w"(_k3n),    // %45
								"w"(_k4n),    // %46
								"w"(_k5n),    // %47
								"w"(_k6n),    // %48
								"w"(_k7n)     // %49
								: "cc", "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25"//, "v26", "v27", "v28", "v29", "v30", "v31"
								);
						}
#else
						for (; nn > 0; nn--)
						{
							float32x4_t _p = vld1q_f32(r0);

							float32x4_t _out0p = vld1q_f32(outptr0);
							float32x4_t _out1p = vld1q_f32(outptr1);
							float32x4_t _out2p = vld1q_f32(outptr2);
							float32x4_t _out3p = vld1q_f32(outptr3);
							float32x4_t _out4p = vld1q_f32(outptr4);
							float32x4_t _out5p = vld1q_f32(outptr5);
							float32x4_t _out6p = vld1q_f32(outptr6);
							float32x4_t _out7p = vld1q_f32(outptr7);

							_out0p = vfmaq_laneq_f32(_out0p, _p, _k0, 0);
							_out1p = vfmaq_laneq_f32(_out1p, _p, _k1, 0);
							_out2p = vfmaq_laneq_f32(_out2p, _p, _k2, 0);
							_out3p = vfmaq_laneq_f32(_out3p, _p, _k3, 0);
							_out4p = vfmaq_laneq_f32(_out4p, _p, _k4, 0);
							_out5p = vfmaq_laneq_f32(_out5p, _p, _k5, 0);
							_out6p = vfmaq_laneq_f32(_out6p, _p, _k6, 0);
							_out7p = vfmaq_laneq_f32(_out7p, _p, _k7, 0);

							float32x4_t _p1 = vld1q_f32(r1);

							_out0p = vfmaq_laneq_f32(_out0p, _p1, _k0, 1);
							_out1p = vfmaq_laneq_f32(_out1p, _p1, _k1, 1);
							_out2p = vfmaq_laneq_f32(_out2p, _p1, _k2, 1);
							_out3p = vfmaq_laneq_f32(_out3p, _p1, _k3, 1);
							_out4p = vfmaq_laneq_f32(_out4p, _p1, _k4, 1);
							_out5p = vfmaq_laneq_f32(_out5p, _p1, _k5, 1);
							_out6p = vfmaq_laneq_f32(_out6p, _p1, _k6, 1);
							_out7p = vfmaq_laneq_f32(_out7p, _p1, _k7, 1);

							float32x4_t _p2 = vld1q_f32(r2);

							_out0p = vfmaq_laneq_f32(_out0p, _p2, _k0, 2);
							_out1p = vfmaq_laneq_f32(_out1p, _p2, _k1, 2);
							_out2p = vfmaq_laneq_f32(_out2p, _p2, _k2, 2);
							_out3p = vfmaq_laneq_f32(_out3p, _p2, _k3, 2);
							_out4p = vfmaq_laneq_f32(_out4p, _p2, _k4, 2);
							_out5p = vfmaq_laneq_f32(_out5p, _p2, _k5, 2);
							_out6p = vfmaq_laneq_f32(_out6p, _p2, _k6, 2);
							_out7p = vfmaq_laneq_f32(_out7p, _p2, _k7, 2);

							float32x4_t _p3 = vld1q_f32(r3);

							_out0p = vfmaq_laneq_f32(_out0p, _p3, _k0, 3);
							_out1p = vfmaq_laneq_f32(_out1p, _p3, _k1, 3);
							_out2p = vfmaq_laneq_f32(_out2p, _p3, _k2, 3);
							_out3p = vfmaq_laneq_f32(_out3p, _p3, _k3, 3);
							_out4p = vfmaq_laneq_f32(_out4p, _p3, _k4, 3);
							_out5p = vfmaq_laneq_f32(_out5p, _p3, _k5, 3);
							_out6p = vfmaq_laneq_f32(_out6p, _p3, _k6, 3);
							_out7p = vfmaq_laneq_f32(_out7p, _p3, _k7, 3);

							float32x4_t _p4 = vld1q_f32(r4);

							_out0p = vfmaq_laneq_f32(_out0p, _p4, _k0n, 0);
							_out1p = vfmaq_laneq_f32(_out1p, _p4, _k1n, 0);
							_out2p = vfmaq_laneq_f32(_out2p, _p4, _k2n, 0);
							_out3p = vfmaq_laneq_f32(_out3p, _p4, _k3n, 0);
							_out4p = vfmaq_laneq_f32(_out4p, _p4, _k4n, 0);
							_out5p = vfmaq_laneq_f32(_out5p, _p4, _k5n, 0);
							_out6p = vfmaq_laneq_f32(_out6p, _p4, _k6n, 0);
							_out7p = vfmaq_laneq_f32(_out7p, _p4, _k7n, 0);

							float32x4_t _p5 = vld1q_f32(r5);

							_out0p = vfmaq_laneq_f32(_out0p, _p5, _k0n, 1);
							_out1p = vfmaq_laneq_f32(_out1p, _p5, _k1n, 1);
							_out2p = vfmaq_laneq_f32(_out2p, _p5, _k2n, 1);
							_out3p = vfmaq_laneq_f32(_out3p, _p5, _k3n, 1);
							_out4p = vfmaq_laneq_f32(_out4p, _p5, _k4n, 1);
							_out5p = vfmaq_laneq_f32(_out5p, _p5, _k5n, 1);
							_out6p = vfmaq_laneq_f32(_out6p, _p5, _k6n, 1);
							_out7p = vfmaq_laneq_f32(_out7p, _p5, _k7n, 1);

							float32x4_t _p6 = vld1q_f32(r6);

							_out0p = vfmaq_laneq_f32(_out0p, _p6, _k0n, 2);
							_out1p = vfmaq_laneq_f32(_out1p, _p6, _k1n, 2);
							_out2p = vfmaq_laneq_f32(_out2p, _p6, _k2n, 2);
							_out3p = vfmaq_laneq_f32(_out3p, _p6, _k3n, 2);
							_out4p = vfmaq_laneq_f32(_out4p, _p6, _k4n, 2);
							_out5p = vfmaq_laneq_f32(_out5p, _p6, _k5n, 2);
							_out6p = vfmaq_laneq_f32(_out6p, _p6, _k6n, 2);
							_out7p = vfmaq_laneq_f32(_out7p, _p6, _k7n, 2);

							float32x4_t _p7 = vld1q_f32(r7);

							_out0p = vfmaq_laneq_f32(_out0p, _p7, _k0n, 3);
							_out1p = vfmaq_laneq_f32(_out1p, _p7, _k1n, 3);
							_out2p = vfmaq_laneq_f32(_out2p, _p7, _k2n, 3);
							_out3p = vfmaq_laneq_f32(_out3p, _p7, _k3n, 3);
							_out4p = vfmaq_laneq_f32(_out4p, _p7, _k4n, 3);
							_out5p = vfmaq_laneq_f32(_out5p, _p7, _k5n, 3);
							_out6p = vfmaq_laneq_f32(_out6p, _p7, _k6n, 3);
							_out7p = vfmaq_laneq_f32(_out7p, _p7, _k7n, 3);

							vst1q_f32(outptr0, _out0p);
							vst1q_f32(outptr1, _out1p);
							vst1q_f32(outptr2, _out2p);
							vst1q_f32(outptr3, _out3p);
							vst1q_f32(outptr4, _out4p);
							vst1q_f32(outptr5, _out5p);
							vst1q_f32(outptr6, _out6p);
							vst1q_f32(outptr7, _out7p);

							r0 += 4;
							r1 += 4;
							r2 += 4;
							r3 += 4;
							r4 += 4;
							r5 += 4;
							r6 += 4;
							r7 += 4;
							outptr0 += 4;
							outptr1 += 4;
							outptr2 += 4;
							outptr3 += 4;
							outptr4 += 4;
							outptr5 += 4;
							outptr6 += 4;
							outptr7 += 4;
						}
#endif
						for (; remain > 0; remain--)
						{
							// TODO neon optimize
							float sum0 = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3] + *r4 * kernel0[4] + *r5 * kernel0[5] + *r6 * kernel0[6] + *r7 * kernel0[7];
							float sum1 = *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3] + *r4 * kernel1[4] + *r5 * kernel1[5] + *r6 * kernel1[6] + *r7 * kernel1[7];
							float sum2 = *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3] + *r4 * kernel2[4] + *r5 * kernel2[5] + *r6 * kernel2[6] + *r7 * kernel2[7];
							float sum3 = *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3] + *r4 * kernel3[4] + *r5 * kernel3[5] + *r6 * kernel3[6] + *r7 * kernel3[7];
							float sum4 = *r0 * kernel4[0] + *r1 * kernel4[1] + *r2 * kernel4[2] + *r3 * kernel4[3] + *r4 * kernel4[4] + *r5 * kernel4[5] + *r6 * kernel4[6] + *r7 * kernel4[7];
							float sum5 = *r0 * kernel5[0] + *r1 * kernel5[1] + *r2 * kernel5[2] + *r3 * kernel5[3] + *r4 * kernel5[4] + *r5 * kernel5[5] + *r6 * kernel5[6] + *r7 * kernel5[7];
							float sum6 = *r0 * kernel6[0] + *r1 * kernel6[1] + *r2 * kernel6[2] + *r3 * kernel6[3] + *r4 * kernel6[4] + *r5 * kernel6[5] + *r6 * kernel6[6] + *r7 * kernel6[7];
							float sum7 = *r0 * kernel7[0] + *r1 * kernel7[1] + *r2 * kernel7[2] + *r3 * kernel7[3] + *r4 * kernel7[4] + *r5 * kernel7[5] + *r6 * kernel7[6] + *r7 * kernel7[7];

							*outptr0 += sum0;
							*outptr1 += sum1;
							*outptr2 += sum2;
							*outptr3 += sum3;
							*outptr4 += sum4;
							*outptr5 += sum5;
							*outptr6 += sum6;
							*outptr7 += sum7;

							r0++;
							r1++;
							r2++;
							r3++;
							r4++;
							r5++;
							r6++;
							r7++;
							outptr0++;
							outptr1++;
							outptr2++;
							outptr3++;
							outptr4++;
							outptr5++;
							outptr6++;
							outptr7++;
						}
					}

					for (; q < inch; q++)
					{
						float* outptr0 = out0;
						float* outptr1 = out1;
						float* outptr2 = out2;
						float* outptr3 = out3;
						float* outptr4 = out4;
						float* outptr5 = out5;
						float* outptr6 = out6;
						float* outptr7 = out7;

						const float* img0 = bottom_data + (q)* bottom_cstep;

						const float* kernel0 = kernel + p * inch + q;
						const float* kernel1 = kernel + (p + 1)*inch + q;
						const float* kernel2 = kernel + (p + 2)*inch + q;
						const float* kernel3 = kernel + (p + 3)*inch + q;
						const float* kernel4 = kernel + (p + 4)*inch + q;
						const float* kernel5 = kernel + (p + 5)*inch + q;
						const float* kernel6 = kernel + (p + 6)*inch + q;
						const float* kernel7 = kernel + (p + 7)*inch + q;

						const float k0 = kernel0[0];
						const float k1 = kernel1[0];
						const float k2 = kernel2[0];
						const float k3 = kernel3[0];
						const float k4 = kernel4[0];
						const float k5 = kernel5[0];
						const float k6 = kernel6[0];
						const float k7 = kernel7[0];

						const float* r0 = img0;

						int size = outw * outh;

						int nn = size >> 2;
						int remain = size & 3;

						float32x4_t _k0 = vdupq_n_f32(k0);
						float32x4_t _k1 = vdupq_n_f32(k1);
						float32x4_t _k2 = vdupq_n_f32(k2);
						float32x4_t _k3 = vdupq_n_f32(k3);
						float32x4_t _k4 = vdupq_n_f32(k4);
						float32x4_t _k5 = vdupq_n_f32(k5);
						float32x4_t _k6 = vdupq_n_f32(k6);
						float32x4_t _k7 = vdupq_n_f32(k7);

						for (; nn > 0; nn--)
						{
							float32x4_t _p = vld1q_f32(r0);

							float32x4_t _out0p = vld1q_f32(outptr0);
							float32x4_t _out1p = vld1q_f32(outptr1);
							float32x4_t _out2p = vld1q_f32(outptr2);
							float32x4_t _out3p = vld1q_f32(outptr3);
							float32x4_t _out4p = vld1q_f32(outptr4);
							float32x4_t _out5p = vld1q_f32(outptr5);
							float32x4_t _out6p = vld1q_f32(outptr6);
							float32x4_t _out7p = vld1q_f32(outptr7);

							_out0p = vfmaq_f32(_out0p, _p, _k0);
							_out1p = vfmaq_f32(_out1p, _p, _k1);
							_out2p = vfmaq_f32(_out2p, _p, _k2);
							_out3p = vfmaq_f32(_out3p, _p, _k3);
							_out4p = vfmaq_f32(_out4p, _p, _k4);
							_out5p = vfmaq_f32(_out5p, _p, _k5);
							_out6p = vfmaq_f32(_out6p, _p, _k6);
							_out7p = vfmaq_f32(_out7p, _p, _k7);

							vst1q_f32(outptr0, _out0p);
							vst1q_f32(outptr1, _out1p);
							vst1q_f32(outptr2, _out2p);
							vst1q_f32(outptr3, _out3p);
							vst1q_f32(outptr4, _out4p);
							vst1q_f32(outptr5, _out5p);
							vst1q_f32(outptr6, _out6p);
							vst1q_f32(outptr7, _out7p);

							r0 += 4;
							outptr0 += 4;
							outptr1 += 4;
							outptr2 += 4;
							outptr3 += 4;
							outptr4 += 4;
							outptr5 += 4;
							outptr6 += 4;
							outptr7 += 4;
						}
						for (; remain > 0; remain--)
						{
							// TODO neon optimize
							float sum0 = *r0 * k0;
							float sum1 = *r0 * k1;
							float sum2 = *r0 * k2;
							float sum3 = *r0 * k3;
							float sum4 = *r0 * k4;
							float sum5 = *r0 * k5;
							float sum6 = *r0 * k6;
							float sum7 = *r0 * k7;

							*outptr0 += sum0;
							*outptr1 += sum1;
							*outptr2 += sum2;
							*outptr3 += sum3;
							*outptr4 += sum4;
							*outptr5 += sum5;
							*outptr6 += sum6;
							*outptr7 += sum7;

							r0++;
							outptr0++;
							outptr1++;
							outptr2++;
							outptr3++;
							outptr4++;
							outptr5++;
							outptr6++;
							outptr7++;
						}
					}
				}

#else

				nn_outch = outch / 6;
				remain_outch_start = nn_outch * 6;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
				for (int pp = 0; pp < nn_outch; pp++)
				{
					int p = pp * 6;

					float *out0 = top_data + (p + 0) * top_cstep;
					float *out1 = top_data + (p + 1) * top_cstep;
					float *out2 = top_data + (p + 2) * top_cstep;
					float *out3 = top_data + (p + 3) * top_cstep;
					float *out4 = top_data + (p + 4) * top_cstep;
					float *out5 = top_data + (p + 5) * top_cstep;

					const float bias0 = this->bias_term_ ? bias[p] : 0.f;
					const float bias1 = this->bias_term_ ? bias[p + 1] : 0.f;
					const float bias2 = this->bias_term_ ? bias[p + 2] : 0.f;
					const float bias3 = this->bias_term_ ? bias[p + 3] : 0.f;
					const float bias4 = this->bias_term_ ? bias[p + 4] : 0.f;
					const float bias5 = this->bias_term_ ? bias[p + 5] : 0.f;

					fill(out0, top_cstep, bias0);
					fill(out1, top_cstep, bias1);
					fill(out2, top_cstep, bias2);
					fill(out3, top_cstep, bias3);
					fill(out4, top_cstep, bias4);
					fill(out5, top_cstep, bias5);

					int q = 0;

					for (; q + 3 < inch; q += 4)
					{
						float* outptr0 = out0;
						float* outptr1 = out1;
						float* outptr2 = out2;
						float* outptr3 = out3;
						float* outptr4 = out4;
						float* outptr5 = out5;

						const float* img0 = bottom_data + (q + 0) * bottom_cstep;
						const float* img1 = bottom_data + (q + 1) * bottom_cstep;
						const float* img2 = bottom_data + (q + 2) * bottom_cstep;
						const float* img3 = bottom_data + (q + 3) * bottom_cstep;

						const float* kernel0 = kernel + p * inch + q;
						const float* kernel1 = kernel + (p + 1)*inch + q;
						const float* kernel2 = kernel + (p + 2)*inch + q;
						const float* kernel3 = kernel + (p + 3)*inch + q;
						const float* kernel4 = kernel + (p + 4)*inch + q;
						const float* kernel5 = kernel + (p + 5)*inch + q;

						const float* r0 = img0;
						const float* r1 = img1;
						const float* r2 = img2;
						const float* r3 = img3;

						int size = outw * outh;

#if __ARM_NEON
						int nn = size >> 2;
						int remain = size & 3;
#else
						int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
						float32x4_t _k0 = vld1q_f32(kernel0);
						float32x4_t _k1 = vld1q_f32(kernel1);
						float32x4_t _k2 = vld1q_f32(kernel2);
						float32x4_t _k3 = vld1q_f32(kernel3);
						float32x4_t _k4 = vld1q_f32(kernel4);
						float32x4_t _k5 = vld1q_f32(kernel5);

						if (nn > 0)
						{
							asm volatile(
								"pld        [%7, #128]              \n"
								"vld1.f32   {d24-d25}, [%7 :128]!   \n"// q12 = r0

								"pld        [%1, #128]              \n"
								"vld1.f32   {d12-d13}, [%1 :128]    \n"// q6 = outptr0

								"pld        [%2, #128]              \n"
								"vld1.f32   {d14-d15}, [%2 :128]    \n"// q7 = outptr1

								"vmla.f32   q6, q12, %e22[0]        \n"

								"0:                                 \n"

								"pld        [%3, #128]              \n"
								"vld1.f32   {d16-d17}, [%3 :128]    \n"// q8 = outptr2

								"vmla.f32   q7, q12, %e23[0]        \n"

								"pld        [%4, #128]              \n"
								"vld1.f32   {d18-d19}, [%4 :128]    \n"// q9 = outptr3

								"vmla.f32   q8, q12, %e24[0]        \n"

								"pld        [%8, #128]              \n"
								"vld1.f32   {d26-d27}, [%8 :128]!   \n"// q13 = r1

								"vmla.f32   q9, q12, %e25[0]        \n"

								"pld        [%5, #128]              \n"
								"vld1.f32   {d20-d21}, [%5 :128]    \n"// q10 = outptr4

								"vmla.f32   q6, q13, %e22[1]        \n"
								"vmla.f32   q7, q13, %e23[1]        \n"

								"pld        [%6, #128]              \n"
								"vld1.f32   {d22-d23}, [%6 :128]    \n"// q11 = outptr5

								"vmla.f32   q10, q12, %e26[0]       \n"
								"vmla.f32   q11, q12, %e27[0]       \n"

								"vmla.f32   q8, q13, %e24[1]        \n"
								"vmla.f32   q9, q13, %e25[1]        \n"

								"pld        [%9, #128]              \n"
								"vld1.f32   {d28-d29}, [%9 :128]!   \n"// q14 = r2

								"vmla.f32   q10, q13, %e26[1]       \n"
								"vmla.f32   q11, q13, %e27[1]       \n"

								"vmla.f32   q6, q14, %f22[0]        \n"
								"vmla.f32   q7, q14, %f23[0]        \n"
								"vmla.f32   q8, q14, %f24[0]        \n"
								"vmla.f32   q9, q14, %f25[0]        \n"

								"pld        [%10, #128]             \n"
								"vld1.f32   {d30-d31}, [%10 :128]!  \n"// q15 = r3

								"vmla.f32   q10, q14, %f26[0]       \n"
								"vmla.f32   q11, q14, %f27[0]       \n"

								"vmla.f32   q6, q15, %f22[1]        \n"
								"vmla.f32   q7, q15, %f23[1]        \n"
								"vmla.f32   q8, q15, %f24[1]        \n"
								"vmla.f32   q9, q15, %f25[1]        \n"

								"pld        [%7, #128]              \n"
								"vld1.f32   {d24-d25}, [%7 :128]!   \n"// q12 = r0

								"vmla.f32   q10, q15, %f26[1]       \n"
								"vmla.f32   q11, q15, %f27[1]       \n"

								"vst1.f32   {d12-d13}, [%1 :128]!   \n"
								"vst1.f32   {d14-d15}, [%2 :128]!   \n"

								"pld        [%1, #128]              \n"
								"vld1.f32   {d12-d13}, [%1 :128]    \n"// q6 = outptr0

								"vst1.f32   {d16-d17}, [%3 :128]!   \n"
								"vst1.f32   {d18-d19}, [%4 :128]!   \n"

								"vmla.f32   q6, q12, %e22[0]        \n"

								"pld        [%2, #128]              \n"
								"vld1.f32   {d14-d15}, [%2 :128]    \n"// q7 = outptr1

								"subs       %0, #1                  \n"

								"vst1.f32   {d20-d21}, [%5 :128]!   \n"
								"vst1.f32   {d22-d23}, [%6 :128]!   \n"

								"bne        0b                      \n"

								"sub        %7, #16                 \n"

								: "=r"(nn),     // %0
								"=r"(outptr0),// %1
								"=r"(outptr1),// %2
								"=r"(outptr2),// %3
								"=r"(outptr3),// %4
								"=r"(outptr4),// %5
								"=r"(outptr5),// %6
								"=r"(r0),     // %7
								"=r"(r1),     // %8
								"=r"(r2),     // %9
								"=r"(r3)      // %10
								: "0"(nn),
								"1"(outptr0),
								"2"(outptr1),
								"3"(outptr2),
								"4"(outptr3),
								"5"(outptr4),
								"6"(outptr5),
								"7"(r0),
								"8"(r1),
								"9"(r2),
								"10"(r3),
								"w"(_k0),     // %22
								"w"(_k1),     // %23
								"w"(_k2),     // %24
								"w"(_k3),     // %25
								"w"(_k4),     // %26
								"w"(_k5)      // %27
								: "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
								);
						}
#endif // __ARM_NEON

						for (; remain > 0; remain--)
						{
							// TODO neon optimize

							float sum0 = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3];
							float sum1 = *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3];
							float sum2 = *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3];
							float sum3 = *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3];
							float sum4 = *r0 * kernel4[0] + *r1 * kernel4[1] + *r2 * kernel4[2] + *r3 * kernel4[3];
							float sum5 = *r0 * kernel5[0] + *r1 * kernel5[1] + *r2 * kernel5[2] + *r3 * kernel5[3];

							*outptr0 += sum0;
							*outptr1 += sum1;
							*outptr2 += sum2;
							*outptr3 += sum3;
							*outptr4 += sum4;
							*outptr5 += sum5;

							r0++;
							r1++;
							r2++;
							r3++;
							outptr0++;
							outptr1++;
							outptr2++;
							outptr3++;
							outptr4++;
							outptr5++;
						}
					}

					for (; q < inch; q++)
					{
						float* outptr0 = out0;
						float* outptr1 = out1;
						float* outptr2 = out2;
						float* outptr3 = out3;
						float* outptr4 = out4;
						float* outptr5 = out5;

						const float* img0 = bottom_data + (q)* bottom_cstep;

						const float* kernel0 = kernel + p * inch + q;
						const float* kernel1 = kernel + (p + 1)*inch + q;
						const float* kernel2 = kernel + (p + 2)*inch + q;
						const float* kernel3 = kernel + (p + 3)*inch + q;
						const float* kernel4 = kernel + (p + 4)*inch + q;
						const float* kernel5 = kernel + (p + 5)*inch + q;

						const float k0 = kernel0[0];
						const float k1 = kernel1[0];
						const float k2 = kernel2[0];
						const float k3 = kernel3[0];
						const float k4 = kernel4[0];
						const float k5 = kernel5[0];

						const float* r0 = img0;

						int size = outw * outh;

#if __ARM_NEON
						int nn = size >> 2;
						int remain = size & 3;
#else
						int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
						float32x4_t _k0 = vdupq_n_f32(k0);
						float32x4_t _k1 = vdupq_n_f32(k1);
						float32x4_t _k2 = vdupq_n_f32(k2);
						float32x4_t _k3 = vdupq_n_f32(k3);
						float32x4_t _k4 = vdupq_n_f32(k4);
						float32x4_t _k5 = vdupq_n_f32(k5);

						if (nn > 0)
						{
							asm volatile(
								"pld        [%7, #128]              \n"
								"vld1.f32   {d24-d25}, [%7 :128]!   \n"// q12 = r0

								"pld        [%1, #128]              \n"
								"vld1.f32   {d12-d13}, [%1 :128]    \n"// q6 = outptr0

								"0:                                 \n"

								"pld        [%2, #128]              \n"
								"vld1.f32   {d14-d15}, [%2 :128]    \n"// q7 = outptr1

								"vmla.f32   q6, q12, %q16           \n"

								"pld        [%3, #128]              \n"
								"vld1.f32   {d16-d17}, [%3 :128]    \n"// q8 = outptr2

								"vmla.f32   q7, q12, %q17           \n"

								"pld        [%4, #128]              \n"
								"vld1.f32   {d18-d19}, [%4 :128]    \n"// q9 = outptr3

								"vmla.f32   q8, q12, %q18           \n"

								"pld        [%5, #128]              \n"
								"vld1.f32   {d20-d21}, [%5 :128]    \n"// q10 = outptr4

								"vmla.f32   q9, q12, %q19           \n"

								"pld        [%6, #128]              \n"
								"vld1.f32   {d22-d23}, [%6 :128]    \n"// q11 = outptr5

								"vmla.f32   q10, q12, %q20          \n"
								"vmla.f32   q11, q12, %q21          \n"

								"pld        [%7, #128]              \n"
								"vld1.f32   {d24-d25}, [%7 :128]!   \n"// q12 = r0

								"vst1.f32   {d12-d13}, [%1 :128]!   \n"
								"vst1.f32   {d14-d15}, [%2 :128]!   \n"

								"pld        [%1, #128]              \n"
								"vld1.f32   {d12-d13}, [%1 :128]    \n"// q6 = outptr0

								"vst1.f32   {d16-d17}, [%3 :128]!   \n"
								"vst1.f32   {d18-d19}, [%4 :128]!   \n"

								"subs       %0, #1                  \n"

								"vst1.f32   {d20-d21}, [%5 :128]!   \n"
								"vst1.f32   {d22-d23}, [%6 :128]!   \n"

								"bne        0b                      \n"

								"sub        %7, #16                 \n"

								: "=r"(nn),     // %0
								"=r"(outptr0),// %1
								"=r"(outptr1),// %2
								"=r"(outptr2),// %3
								"=r"(outptr3),// %4
								"=r"(outptr4),// %5
								"=r"(outptr5),// %6
								"=r"(r0)      // %7
								: "0"(nn),
								"1"(outptr0),
								"2"(outptr1),
								"3"(outptr2),
								"4"(outptr3),
								"5"(outptr4),
								"6"(outptr5),
								"7"(r0),
								"w"(_k0),     // %16
								"w"(_k1),     // %17
								"w"(_k2),     // %18
								"w"(_k3),     // %19
								"w"(_k4),     // %20
								"w"(_k5)      // %21
								: "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12"
								);
						}
#endif // __ARM_NEON
						for (; remain > 0; remain--)
						{
							// TODO neon optimize
							float sum0 = *r0 * k0;
							float sum1 = *r0 * k1;
							float sum2 = *r0 * k2;
							float sum3 = *r0 * k3;
							float sum4 = *r0 * k4;
							float sum5 = *r0 * k5;

							*outptr0 += sum0;
							*outptr1 += sum1;
							*outptr2 += sum2;
							*outptr3 += sum3;
							*outptr4 += sum4;
							*outptr5 += sum5;

							r0++;
							outptr0++;
							outptr1++;
							outptr2++;
							outptr3++;
							outptr4++;
							outptr5++;
						}
					}
				}
#endif // __ARM_NEON && __aarch64__

				nn_outch = (outch - remain_outch_start) >> 2;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
				for (int pp = 0; pp < nn_outch; pp++)
				{
					int p = remain_outch_start + pp * 4;

					float *out0 = top_data + (p)* top_cstep;
					float *out1 = top_data + (p + 1) * top_cstep;
					float *out2 = top_data + (p + 2) * top_cstep;
					float *out3 = top_data + (p + 3) * top_cstep;

					const float bias0 = bias ? bias[p] : 0.f;
					const float bias1 = bias ? bias[p + 1] : 0.f;
					const float bias2 = bias ? bias[p + 2] : 0.f;
					const float bias3 = bias ? bias[p + 3] : 0.f;

					fill(out0, top_cstep, bias0);
					fill(out1, top_cstep, bias1);
					fill(out2, top_cstep, bias2);
					fill(out3, top_cstep, bias3);

					int q = 0;

					for (; q + 3 < inch; q += 4)
					{
						float* outptr0 = out0;
						float* outptr1 = out1;
						float* outptr2 = out2;
						float* outptr3 = out3;

						const float* img0 = bottom_data + (q + 0) * bottom_cstep;
						const float* img1 = bottom_data + (q + 1) * bottom_cstep;
						const float* img2 = bottom_data + (q + 2) * bottom_cstep;
						const float* img3 = bottom_data + (q + 3) * bottom_cstep;

						const float* kernel0 = kernel + p * inch + q;
						const float* kernel1 = kernel + (p + 1)*inch + q;
						const float* kernel2 = kernel + (p + 2)*inch + q;
						const float* kernel3 = kernel + (p + 3)*inch + q;

						const float* r0 = img0;
						const float* r1 = img1;
						const float* r2 = img2;
						const float* r3 = img3;

						int size = outw * outh;

#if __ARM_NEON
						int nn = size >> 3;
						int remain = size & 7;
#else
						int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
						float32x4_t _k0 = vld1q_f32(kernel0);
						float32x4_t _k1 = vld1q_f32(kernel1);
						float32x4_t _k2 = vld1q_f32(kernel2);
						float32x4_t _k3 = vld1q_f32(kernel3);

#if __aarch64__
						if (nn > 0)
						{
							asm volatile(
								"prfm   pldl1keep, [%5, #256]       \n"
								"ld1    {v6.4s, v7.4s}, [%5], #32   \n"

								"prfm   pldl1keep, [%1, #256]       \n"
								"ld1    {v8.4s, v9.4s}, [%1]        \n"

								"0:                                 \n"

								"fmla   v8.4s, v6.4s, %18.s[0]      \n"

								"prfm   pldl1keep, [%2, #256]       \n"
								"ld1    {v10.4s, v11.4s}, [%2]      \n"

								"fmla   v9.4s, v7.4s, %18.s[0]      \n"

								"fmla   v10.4s, v6.4s, %19.s[0]     \n"

								"prfm   pldl1keep, [%3, #256]       \n"
								"ld1    {v12.4s, v13.4s}, [%3]      \n"

								"fmla   v11.4s, v7.4s, %19.s[0]     \n"

								"fmla   v12.4s, v6.4s, %20.s[0]     \n"

								"prfm   pldl1keep, [%4, #256]       \n"
								"ld1    {v14.4s, v15.4s}, [%4]      \n"

								"fmla   v13.4s, v7.4s, %20.s[0]     \n"

								"prfm   pldl1keep, [%6, #256]       \n"
								"ld1    {v4.4s, v5.4s}, [%6], #32   \n"

								"fmla   v14.4s, v6.4s, %21.s[0]     \n"
								"fmla   v15.4s, v7.4s, %21.s[0]     \n"

								"fmla   v8.4s, v4.4s, %18.s[1]      \n"
								"fmla   v9.4s, v5.4s, %18.s[1]      \n"

								"fmla   v10.4s, v4.4s, %19.s[1]     \n"
								"fmla   v11.4s, v5.4s, %19.s[1]     \n"

								"fmla   v12.4s, v4.4s, %20.s[1]     \n"
								"fmla   v13.4s, v5.4s, %20.s[1]     \n"

								"prfm   pldl1keep, [%7, #256]       \n"
								"ld1    {v6.4s, v7.4s}, [%7], #32   \n"

								"fmla   v14.4s, v4.4s, %21.s[1]     \n"
								"fmla   v15.4s, v5.4s, %21.s[1]     \n"

								"fmla   v8.4s, v6.4s, %18.s[2]      \n"
								"fmla   v9.4s, v7.4s, %18.s[2]      \n"

								"fmla   v10.4s, v6.4s, %19.s[2]     \n"
								"fmla   v11.4s, v7.4s, %19.s[2]     \n"

								"fmla   v12.4s, v6.4s, %20.s[2]     \n"
								"fmla   v13.4s, v7.4s, %20.s[2]     \n"

								"prfm   pldl1keep, [%8, #256]       \n"
								"ld1    {v4.4s, v5.4s}, [%8], #32   \n"

								"fmla   v14.4s, v6.4s, %21.s[2]     \n"
								"fmla   v15.4s, v7.4s, %21.s[2]     \n"

								"fmla   v8.4s, v4.4s, %18.s[3]      \n"
								"fmla   v9.4s, v5.4s, %18.s[3]      \n"

								"fmla   v10.4s, v4.4s, %19.s[3]     \n"
								"fmla   v11.4s, v5.4s, %19.s[3]     \n"

								"st1    {v8.4s, v9.4s}, [%1], #32   \n"

								"fmla   v12.4s, v4.4s, %20.s[3]     \n"
								"fmla   v13.4s, v5.4s, %20.s[3]     \n"

								"st1    {v10.4s, v11.4s}, [%2], #32 \n"

								"prfm   pldl1keep, [%5, #256]       \n"
								"ld1    {v6.4s, v7.4s}, [%5], #32   \n"

								"fmla   v14.4s, v4.4s, %21.s[3]     \n"
								"fmla   v15.4s, v5.4s, %21.s[3]     \n"

								"st1    {v12.4s, v13.4s}, [%3], #32 \n"

								"prfm   pldl1keep, [%1, #256]       \n"
								"ld1    {v8.4s, v9.4s}, [%1]        \n"

								"subs   %w0, %w0, #1                \n"

								"st1    {v14.4s, v15.4s}, [%4], #32 \n"

								"bne    0b                          \n"
								"sub    %5, %5, #32                 \n"
								: "=r"(nn),     // %0
								"=r"(outptr0),// %1
								"=r"(outptr1),// %2
								"=r"(outptr2),// %3
								"=r"(outptr3),// %4
								"=r"(r0),     // %5
								"=r"(r1),     // %6
								"=r"(r2),     // %7
								"=r"(r3)      // %8
								: "0"(nn),
								"1"(outptr0),
								"2"(outptr1),
								"3"(outptr2),
								"4"(outptr3),
								"5"(r0),
								"6"(r1),
								"7"(r2),
								"8"(r3),
								"w"(_k0),     // %18
								"w"(_k1),     // %19
								"w"(_k2),     // %20
								"w"(_k3)      // %21
								: "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
								);
						}
#else
						if (nn > 0)
						{
							asm volatile(
								"pld        [%5, #256]              \n"
								"vld1.f32   {d12-d15}, [%5 :128]!   \n"
								"pld        [%1, #256]              \n"
								"vld1.f32   {d16-d19}, [%1 :128]    \n"
								"0:                                 \n"

								"vmla.f32   q8, q6, %e18[0]         \n"

								"pld        [%2, #256]              \n"
								"vld1.f32   {d20-d23}, [%2 :128]    \n"
								"vmla.f32   q9, q7, %e18[0]         \n"

								"vmla.f32   q10, q6, %e19[0]        \n"

								"pld        [%3, #256]              \n"
								"vld1.f32   {d24-d27}, [%3 :128]    \n"
								"vmla.f32   q11, q7, %e19[0]        \n"

								"vmla.f32   q12, q6, %e20[0]        \n"

								"pld        [%4, #256]              \n"
								"vld1.f32   {d28-d31}, [%4 :128]    \n"
								"vmla.f32   q13, q7, %e20[0]        \n"

								"pld        [%6, #256]              \n"
								"vld1.f32   {d8-d11}, [%6 :128]!    \n"

								"vmla.f32   q14, q6, %e21[0]        \n"
								"vmla.f32   q15, q7, %e21[0]        \n"

								"vmla.f32   q8, q4, %e18[1]         \n"
								"vmla.f32   q9, q5, %e18[1]         \n"

								"vmla.f32   q10, q4, %e19[1]        \n"
								"vmla.f32   q11, q5, %e19[1]        \n"

								"vmla.f32   q12, q4, %e20[1]        \n"
								"vmla.f32   q13, q5, %e20[1]        \n"

								"pld        [%7, #256]              \n"
								"vld1.f32   {d12-d15}, [%7 :128]!   \n"

								"vmla.f32   q14, q4, %e21[1]        \n"
								"vmla.f32   q15, q5, %e21[1]        \n"

								"vmla.f32   q8, q6, %f18[0]         \n"
								"vmla.f32   q9, q7, %f18[0]         \n"

								"vmla.f32   q10, q6, %f19[0]        \n"
								"vmla.f32   q11, q7, %f19[0]        \n"

								"vmla.f32   q12, q6, %f20[0]        \n"
								"vmla.f32   q13, q7, %f20[0]        \n"

								"pld        [%8, #256]              \n"
								"vld1.f32   {d8-d11}, [%8 :128]!    \n"

								"vmla.f32   q14, q6, %f21[0]        \n"
								"vmla.f32   q15, q7, %f21[0]        \n"

								"vmla.f32   q8, q4, %f18[1]         \n"
								"vmla.f32   q9, q5, %f18[1]         \n"

								"vmla.f32   q10, q4, %f19[1]        \n"
								"vmla.f32   q11, q5, %f19[1]        \n"

								"vmla.f32   q12, q4, %f20[1]        \n"
								"vst1.f32   {d16-d19}, [%1 :128]!   \n"

								"vmla.f32   q13, q5, %f20[1]        \n"

								"vst1.f32   {d20-d23}, [%2 :128]!   \n"

								"vmla.f32   q14, q4, %f21[1]        \n"
								"pld        [%5, #256]              \n"
								"vld1.f32   {d12-d15}, [%5 :128]!   \n"

								"vmla.f32   q15, q5, %f21[1]        \n"

								"vst1.f32   {d24-d27}, [%3 :128]!   \n"

								"pld        [%1, #256]              \n"
								"vld1.f32   {d16-d19}, [%1 :128]    \n"

								"subs       %0, #1                  \n"
								"vst1.f32   {d28-d31}, [%4 :128]!   \n"

								"bne        0b                      \n"
								"sub        %5, #32                 \n"
								: "=r"(nn),     // %0
								"=r"(outptr0),// %1
								"=r"(outptr1),// %2
								"=r"(outptr2),// %3
								"=r"(outptr3),// %4
								"=r"(r0),     // %5
								"=r"(r1),     // %6
								"=r"(r2),     // %7
								"=r"(r3)      // %8
								: "0"(nn),
								"1"(outptr0),
								"2"(outptr1),
								"3"(outptr2),
								"4"(outptr3),
								"5"(r0),
								"6"(r1),
								"7"(r2),
								"8"(r3),
								"w"(_k0),     // %18
								"w"(_k1),     // %19
								"w"(_k2),     // %20
								"w"(_k3)      // %21
								: "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
								);
						}
#endif // __aarch64__
#endif // __ARM_NEON
						for (; remain > 0; remain--)
						{
							// TODO neon optimize
							float sum0 = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3];
							float sum1 = *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3];
							float sum2 = *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3];
							float sum3 = *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3];

							*outptr0 += sum0;
							*outptr1 += sum1;
							*outptr2 += sum2;
							*outptr3 += sum3;

							r0++;
							r1++;
							r2++;
							r3++;
							outptr0++;
							outptr1++;
							outptr2++;
							outptr3++;
						}
					}

					for (; q < inch; q++)
					{
						float* outptr0 = out0;
						float* outptr1 = out1;
						float* outptr2 = out2;
						float* outptr3 = out3;

						const float* img0 = bottom_data + (q)* bottom_cstep;

						const float* kernel0 = kernel + p * inch + q;
						const float* kernel1 = kernel + (p + 1)*inch + q;
						const float* kernel2 = kernel + (p + 2)*inch + q;
						const float* kernel3 = kernel + (p + 3)*inch + q;

						const float k0 = kernel0[0];
						const float k1 = kernel1[0];
						const float k2 = kernel2[0];
						const float k3 = kernel3[0];

						const float* r0 = img0;

						int size = outw * outh;

#if __ARM_NEON
						int nn = size >> 3;
						int remain = size & 7;
#else
						int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
						float32x4_t _k0 = vdupq_n_f32(k0);
						float32x4_t _k1 = vdupq_n_f32(k1);
						float32x4_t _k2 = vdupq_n_f32(k2);
						float32x4_t _k3 = vdupq_n_f32(k3);
#if __aarch64__
						if (nn > 0)
						{
							asm volatile(
								"prfm       pldl1keep, [%5, #256]          \n"
								"ld1        {v6.4s, v7.4s}, [%5], #32      \n"
								"0:                                        \n"
								"prfm       pldl1keep, [%1, #256]          \n"
								"ld1        {v8.4s, v9.4s}, [%1]           \n"
								"fmla       v8.4s, v6.4s, %12.4s           \n"
								"fmla       v9.4s, v7.4s, %12.4s           \n"

								"prfm       pldl1keep, [%2, #256]          \n"
								"ld1        {v10.4s, v11.4s}, [%2]         \n"
								"fmla       v10.4s, v6.4s, %13.4s          \n"
								"fmla       v11.4s, v7.4s, %13.4s          \n"

								"st1        {v8.4s, v9.4s}, [%1], #32      \n"

								"prfm       pldl1keep, [%3, #256]          \n"
								"ld1        {v12.4s, v13.4s}, [%3]         \n"
								"fmla       v12.4s, v6.4s, %14.4s          \n"
								"fmla       v13.4s, v7.4s, %14.4s          \n"

								"st1        {v10.4s, v11.4s}, [%2], #32    \n"

								"prfm       pldl1keep, [%4, #256]          \n"
								"ld1        {v14.4s, v15.4s}, [%4]         \n"
								"fmla       v14.4s, v6.4s, %15.4s          \n"
								"fmla       v15.4s, v7.4s, %15.4s          \n"

								"st1        {v12.4s, v13.4s}, [%3], #32    \n"

								"prfm       pldl1keep, [%5, #256]          \n"
								"ld1        {v6.4s, v7.4s}, [%5], #32      \n"
								"subs       %w0, %w0, #1                   \n"
								"st1        {v14.4s, v15.4s}, [%4], #32    \n"
								"bne        0b                             \n"
								"sub        %5, %5, #32                    \n"
								: "=r"(nn),     // %0
								"=r"(outptr0),// %1
								"=r"(outptr1),// %2
								"=r"(outptr2),// %3
								"=r"(outptr3),// %4
								"=r"(r0)      // %5
								: "0"(nn),
								"1"(outptr0),
								"2"(outptr1),
								"3"(outptr2),
								"4"(outptr3),
								"5"(r0),
								"w"(_k0),     // %12
								"w"(_k1),     // %13
								"w"(_k2),     // %14
								"w"(_k3)      // %15
								: "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
								);
						}
#else
						if (nn > 0)
						{
							asm volatile(
								"pld        [%5, #256]              \n"
								"vld1.f32   {d12-d15}, [%5 :128]!   \n"
								"0:                                 \n"
								"pld        [%1, #256]              \n"
								"vld1.f32   {d16-d19}, [%1 :128]    \n"
								"vmla.f32   q8, q6, %q12            \n"
								"vmla.f32   q9, q7, %q12            \n"

								"pld        [%2, #256]              \n"
								"vld1.f32   {d20-d23}, [%2 :128]    \n"
								"vmla.f32   q10, q6, %q13           \n"
								"vmla.f32   q11, q7, %q13           \n"

								"vst1.f32   {d16-d19}, [%1 :128]!   \n"

								"pld        [%3, #256]              \n"
								"vld1.f32   {d24-d27}, [%3 :128]    \n"
								"vmla.f32   q12, q6, %q14           \n"
								"vmla.f32   q13, q7, %q14           \n"

								"vst1.f32   {d20-d23}, [%2 :128]!   \n"

								"pld        [%4, #256]              \n"
								"vld1.f32   {d28-d31}, [%4 :128]    \n"
								"vmla.f32   q14, q6, %q15           \n"
								"vmla.f32   q15, q7, %q15           \n"

								"vst1.f32   {d24-d27}, [%3 :128]!   \n"

								"pld        [%5, #256]              \n"
								"vld1.f32   {d12-d15}, [%5 :128]!   \n"
								"subs       %0, #1                  \n"
								"vst1.f32   {d28-d31}, [%4 :128]!   \n"
								"bne        0b                      \n"
								"sub        %5, #32                 \n"
								: "=r"(nn),     // %0
								"=r"(outptr0),// %1
								"=r"(outptr1),// %2
								"=r"(outptr2),// %3
								"=r"(outptr3),// %4
								"=r"(r0)      // %5
								: "0"(nn),
								"1"(outptr0),
								"2"(outptr1),
								"3"(outptr2),
								"4"(outptr3),
								"5"(r0),
								"w"(_k0),     // %12
								"w"(_k1),     // %13
								"w"(_k2),     // %14
								"w"(_k3)      // %15
								: "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
								);
						}
#endif // __aarch64__
#endif // __ARM_NEON
						for (; remain > 0; remain--)
						{
							float sum0 = *r0 * k0;
							float sum1 = *r0 * k1;
							float sum2 = *r0 * k2;
							float sum3 = *r0 * k3;

							*outptr0 += sum0;
							*outptr1 += sum1;
							*outptr2 += sum2;
							*outptr3 += sum3;

							r0++;
							outptr0++;
							outptr1++;
							outptr2++;
							outptr3++;
						}
					}
				}

				remain_outch_start += nn_outch << 2;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
				for (int p = remain_outch_start; p < outch; p++)
				{
					float *out = top_data + (p)* top_cstep;

					const float bias0 = bias ? bias[p] : 0.f;

					fill(out, top_cstep, bias0);

					int q = 0;

					for (; q + 3 < inch; q += 4)
					{
						float* outptr = out;

						const float* img0 = bottom_data + (q + 0) * bottom_cstep;
						const float* img1 = bottom_data + (q + 1) * bottom_cstep;
						const float* img2 = bottom_data + (q + 2) * bottom_cstep;
						const float* img3 = bottom_data + (q + 3) * bottom_cstep;

						const float* kernel0 = kernel + p * inch + q;
						const float k0 = kernel0[0];
						const float k1 = kernel0[1];
						const float k2 = kernel0[2];
						const float k3 = kernel0[3];

						const float* r0 = img0;
						const float* r1 = img1;
						const float* r2 = img2;
						const float* r3 = img3;

						int size = outw * outh;

#if __ARM_NEON
						int nn = size >> 3;
						int remain = size & 7;
#else
						int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
						float32x4_t _k0 = vdupq_n_f32(k0);
						float32x4_t _k1 = vdupq_n_f32(k1);
						float32x4_t _k2 = vdupq_n_f32(k2);
						float32x4_t _k3 = vdupq_n_f32(k3);
#if __aarch64__
						if (nn > 0)
						{
							asm volatile(
								"prfm       pldl1keep, [%2, #256]          \n"
								"ld1        {v2.4s, v3.4s}, [%2], #32      \n"
								"0:                                        \n"
								"prfm       pldl1keep, [%1, #256]          \n"
								"ld1        {v0.4s, v1.4s}, [%1]           \n"
								"fmla       v0.4s, v2.4s, %12.4s           \n"
								"fmla       v1.4s, v3.4s, %12.4s           \n"

								"prfm       pldl1keep, [%3, #256]          \n"
								"ld1        {v2.4s, v3.4s}, [%3], #32      \n"
								"fmla       v0.4s, v2.4s, %13.4s           \n"
								"fmla       v1.4s, v3.4s, %13.4s           \n"

								"prfm       pldl1keep, [%4, #256]          \n"
								"ld1        {v2.4s, v3.4s}, [%4], #32      \n"
								"fmla       v0.4s, v2.4s, %14.4s           \n"
								"fmla       v1.4s, v3.4s, %14.4s           \n"

								"prfm       pldl1keep, [%5, #256]          \n"
								"ld1        {v2.4s, v3.4s}, [%5], #32      \n"
								"fmla       v0.4s, v2.4s, %15.4s           \n"
								"fmla       v1.4s, v3.4s, %15.4s           \n"

								"prfm       pldl1keep, [%2, #256]          \n"
								"ld1        {v2.4s, v3.4s}, [%2], #32      \n"
								"subs       %w0, %w0, #1                   \n"
								"st1        {v0.4s, v1.4s}, [%1], #32      \n"
								"bne        0b                             \n"
								"sub        %2, %2, #32                    \n"
								: "=r"(nn),     // %0
								"=r"(outptr), // %1
								"=r"(r0),     // %2
								"=r"(r1),     // %3
								"=r"(r2),     // %4
								"=r"(r3)      // %5
								: "0"(nn),
								"1"(outptr),
								"2"(r0),
								"3"(r1),
								"4"(r2),
								"5"(r3),
								"w"(_k0),     // %12
								"w"(_k1),     // %13
								"w"(_k2),     // %14
								"w"(_k3)      // %15
								: "cc", "memory", "v0", "v1", "v2", "v3"
								);
						}
#else
						if (nn > 0)
						{
							asm volatile(
								"pld        [%2, #256]          \n"
								"vld1.f32   {d4-d7}, [%2 :128]! \n"
								"0:                             \n"
								"pld        [%1, #256]          \n"
								"vld1.f32   {d0-d3}, [%1 :128]  \n"
								"vmla.f32   q0, q2, %q12        \n"
								"vmla.f32   q1, q3, %q12        \n"
								"pld        [%3, #256]          \n"
								"vld1.f32   {d4-d7}, [%3 :128]! \n"
								"vmla.f32   q0, q2, %q13        \n"
								"vmla.f32   q1, q3, %q13        \n"
								"pld        [%4, #256]          \n"
								"vld1.f32   {d4-d7}, [%4 :128]! \n"
								"vmla.f32   q0, q2, %q14        \n"
								"vmla.f32   q1, q3, %q14        \n"
								"pld        [%5, #256]          \n"
								"vld1.f32   {d4-d7}, [%5 :128]! \n"
								"vmla.f32   q0, q2, %q15        \n"
								"vmla.f32   q1, q3, %q15        \n"
								"pld        [%2, #256]          \n"
								"vld1.f32   {d4-d7}, [%2 :128]! \n"
								"subs       %0, #1              \n"
								"vst1.f32   {d0-d3}, [%1 :128]! \n"
								"bne        0b                  \n"
								"sub        %2, #32             \n"
								: "=r"(nn),     // %0
								"=r"(outptr), // %1
								"=r"(r0),     // %2
								"=r"(r1),     // %3
								"=r"(r2),     // %4
								"=r"(r3)      // %5
								: "0"(nn),
								"1"(outptr),
								"2"(r0),
								"3"(r1),
								"4"(r2),
								"5"(r3),
								"w"(_k0),     // %12
								"w"(_k1),     // %13
								"w"(_k2),     // %14
								"w"(_k3)      // %15
								: "cc", "memory", "q0", "q1", "q2", "q3"
								);
						}
#endif // __aarch64__
#endif // __ARM_NEON
						for (; remain > 0; remain--)
						{
							float sum = *r0 * k0;
							float sum1 = *r1 * k1;
							float sum2 = *r2 * k2;
							float sum3 = *r3 * k3;

							*outptr += sum + sum1 + sum2 + sum3;

							r0++;
							r1++;
							r2++;
							r3++;
							outptr++;
						}
					}

					for (; q < inch; q++)
					{
						float* outptr = out;

						const float* img0 = bottom_data + (q)* bottom_cstep;

						const float* kernel0 = kernel + p * inch + q;
						const float k0 = kernel0[0];

						const float* r0 = img0;

						int size = outw * outh;

#if __ARM_NEON
						int nn = size >> 3;
						int remain = size & 7;
#else
						int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
						float32x4_t _k0 = vdupq_n_f32(k0);
#if __aarch64__
						if (nn > 0)
						{
							asm volatile(
								"prfm       pldl1keep, [%2, #256]          \n"
								"ld1        {v2.4s, v3.4s}, [%2], #32      \n"
								"0:                                        \n"
								"prfm       pldl1keep, [%1, #256]          \n"
								"ld1        {v0.4s, v1.4s}, [%1]           \n"
								"fmla       v0.4s, v2.4s, %6.4s            \n"
								"fmla       v1.4s, v3.4s, %6.4s            \n"
								"prfm       pldl1keep, [%2, #256]          \n"
								"ld1        {v2.4s, v3.4s}, [%2], #32      \n"
								"subs       %w0, %w0, #1                   \n"
								"st1        {v0.4s, v1.4s}, [%1], #32      \n"
								"bne        0b                             \n"
								"sub        %2, %2, #32                    \n"
								: "=r"(nn),     // %0
								"=r"(outptr), // %1
								"=r"(r0)      // %2
								: "0"(nn),
								"1"(outptr),
								"2"(r0),
								"w"(_k0)      // %6
								: "cc", "memory", "v0", "v1", "v2", "v3"
								);
						}
#else
						if (nn > 0)
						{
							asm volatile(
								"pld        [%2, #256]          \n"
								"vld1.f32   {d4-d7}, [%2 :128]! \n"
								"0:                             \n"
								"pld        [%1, #256]          \n"
								"vld1.f32   {d0-d3}, [%1 :128]  \n"
								"vmla.f32   q0, q2, %q6         \n"
								"vmla.f32   q1, q3, %q6         \n"
								"pld        [%2, #256]          \n"
								"vld1.f32   {d4-d7}, [%2 :128]! \n"
								"subs       %0, #1              \n"
								"vst1.f32   {d0-d3}, [%1 :128]! \n"
								"bne        0b                  \n"
								"sub        %2, #32             \n"
								: "=r"(nn),     // %0
								"=r"(outptr), // %1
								"=r"(r0)      // %2
								: "0"(nn),
								"1"(outptr),
								"2"(r0),
								"w"(_k0)      // %6
								: "cc", "memory", "q0", "q1", "q2", "q3"
								);
						}
#endif // __aarch64__
#endif // __ARM_NEON
						for (; remain > 0; remain--)
						{
							float sum = *r0 * k0;

							*outptr += sum;

							r0++;
							outptr++;
						}
					}
				}
			}
		}

		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv3x3s1_winograd64_neon5(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top)
		{
			const float *kernel_tm_data = kernel_tm_->cpu_data();
			int kernel_tm_w = kernel_tm_->width();
			int kernel_tm_h = kernel_tm_->height();
			int kernel_tm_cstep = kernel_tm_w * kernel_tm_h;

			int num = bottom->num();
			int w = bottom->width();
			int h = bottom->height();
			int inch = bottom->channels();

			int outw = top->width();
			int outh = top->height();
			int outch = top->channels();

			// pad to 6n+2
			std::shared_ptr<memory::tensor<float> > bottom_bordered;

			outw = (outw + 5) / 6 * 6;
			outh = (outh + 5) / 6 * 6;

			w = outw + 2;
			h = outh + 2;
			int bottom_bordered_cstep = w * h;

			make_border(bottom, bottom_bordered, 0, h - bottom->height(), 0, w - bottom->width());

			const float* bias = nullptr;
			if (this->bias_term_)
				bias = this->weights_f32_[1]->cpu_data();

			std::shared_ptr<memory::tensor<float>> top_bordered(new memory::tensor<float>(std::vector<int>{num, outch, outh, outw}, -1, memory::NCHW, nullptr));
			int top_bordered_w = top_bordered->width();
			int top_bordered_h = top_bordered->height();
			int top_borderd_cstep = top_bordered_w * top_bordered_h;

			int w_tm = outw / 6 * 8;
			int h_tm = outh / 6 * 8;
			const int tiles = w_tm / 8 * h_tm / 8;
			memory::tensor<float> bottom_tm = memory::tensor<float>(std::vector<int>{1, inch, 64 * tiles, 1}, -1, memory::NCHW);
			float *bottom_tm_data = bottom_tm.mutable_cpu_data();
			int bottom_tm_w = bottom_tm.width();
			int bottom_tm_h = bottom_tm.height();
			int bottom_tm_cstep = bottom_tm_w * bottom_tm_h;

			memory::tensor<float> bottom_tm2(std::vector<int>{1, 64, tiles / 8 + (tiles % 8) / 4 + tiles % 4, 8 * inch}, -1, memory::NCHW);
			float *bottom_tm2_data = bottom_tm2.mutable_cpu_data();
			int bottom_tm2_w = bottom_tm2.width();
			int bottom_tm2_h = bottom_tm2.height();
			int bottom_tm2_cstep = bottom_tm2_w * bottom_tm2_h;

			memory::tensor<float> top_tm = memory::tensor<float>(std::vector<int>{1, outch, 64 * tiles, 1}, -1, memory::NCHW);
			int top_tm_w = top_tm.width();
			int top_tm_h = top_tm.height();
			int top_tm_cstep = top_tm_w * top_tm_h;
			float *top_tm_data = top_tm.mutable_cpu_data();

			for (int num_i = 0; num_i < num; num_i++)
			{
				const float *bottom_bordered_data = bottom_bordered->cpu_data() + num_i * inch * bottom_bordered_cstep;

				// BEGIN transform input
				{

					//         bottom_tm.create(inch, tiles, 64);

					//         const float itm[8][8] = {
					//             {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
					//
					//             {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
					//             {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
					//
					//             {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
					//             {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
					//
					//             {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
					//             {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
					//
					//             {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
					//         };

					// 0 = r00 - r06 + (r04 - r02) * 5.25
					// 7 = r07 - r01 + (r03 - r05) * 5.25

					// 1 = (r02 + r06 - r04 * 4.25) + (r01 - r03 * 4.25 + r05)
					// 2 = (r02 + r06 - r04 * 4.25) - (r01 - r03 * 4.25 + r05)

					// 3 = (r06 + r02 * 0.25 - r04 * 1.25) + (r01 * 0.5 - r03 * 2.5 + r05 * 2)
					// 4 = (r06 + r02 * 0.25 - r04 * 1.25) - (r01 * 0.5 - r03 * 2.5 + r05 * 2)

					// reuse r04 * 1.25
					// reuse r03 * 2.5
					// 5 = (r06 + (r02 - r04 * 1.25) * 4) + (r01 * 2 - r03 * 2.5 + r05 * 0.5)
					// 6 = (r06 + (r02 - r04 * 1.25) * 4) - (r01 * 2 - r03 * 2.5 + r05 * 0.5)

#if __ARM_NEON
					const float coeff[8] = {
						0.25f, 0.5f, -1.25f,   2.f,
						-2.5f,  4.f,  4.25f, 5.25f
					};
					float32x4_t _coeff0 = vld1q_f32(coeff);
					float32x4_t _coeff1 = vld1q_f32(coeff + 4);
#endif // __ARM_NEON

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
					for (int q = 0; q < inch; q++)
					{
						const float *img0 = bottom_bordered_data + (q)* bottom_bordered_cstep;
						float *img0_tm = bottom_tm_data + (q)* bottom_tm_cstep;

						float tmp[8][8];

						// tile
						for (int i = 0; i < h_tm / 8; i++)
						{
							for (int j = 0; j < w_tm / 8; j++)
							{
#if __ARM_NEON
								const float* r0 = img0 + (i * 6) * w + j * 6;
								const float* r1 = r0 + w;
								const float* r2 = r0 + w * 2;
								const float* r3 = r0 + w * 3;

#if __aarch64__
								for (int m = 0; m + 3 < 8; m += 4)
								{
									float32x4_t _r0_0123 = vld1q_f32(r0);
									float32x4_t _r0_4567 = vld1q_f32(r0 + 4);
									float32x4_t _r1_0123 = vld1q_f32(r1);
									float32x4_t _r1_4567 = vld1q_f32(r1 + 4);
									float32x4_t _r2_0123 = vld1q_f32(r2);
									float32x4_t _r2_4567 = vld1q_f32(r2 + 4);
									float32x4_t _r3_0123 = vld1q_f32(r3);
									float32x4_t _r3_4567 = vld1q_f32(r3 + 4);

									float32x4x2_t _r01_00221133 = vtrnq_f32(_r0_0123, _r1_0123);
									float32x4x2_t _r01_44665577 = vtrnq_f32(_r0_4567, _r1_4567);
									float32x4x2_t _r23_00221133 = vtrnq_f32(_r2_0123, _r3_0123);
									float32x4x2_t _r23_44665577 = vtrnq_f32(_r2_4567, _r3_4567);

									// no vswp intrinsic  :(
									float32x4_t _r_00 = vcombine_f32(vget_low_f32(_r01_00221133.val[0]), vget_low_f32(_r23_00221133.val[0]));
									float32x4_t _r_11 = vcombine_f32(vget_low_f32(_r01_00221133.val[1]), vget_low_f32(_r23_00221133.val[1]));
									float32x4_t _r_22 = vcombine_f32(vget_high_f32(_r01_00221133.val[0]), vget_high_f32(_r23_00221133.val[0]));
									float32x4_t _r_33 = vcombine_f32(vget_high_f32(_r01_00221133.val[1]), vget_high_f32(_r23_00221133.val[1]));
									float32x4_t _r_44 = vcombine_f32(vget_low_f32(_r01_44665577.val[0]), vget_low_f32(_r23_44665577.val[0]));
									float32x4_t _r_55 = vcombine_f32(vget_low_f32(_r01_44665577.val[1]), vget_low_f32(_r23_44665577.val[1]));
									float32x4_t _r_66 = vcombine_f32(vget_high_f32(_r01_44665577.val[0]), vget_high_f32(_r23_44665577.val[0]));
									float32x4_t _r_77 = vcombine_f32(vget_high_f32(_r01_44665577.val[1]), vget_high_f32(_r23_44665577.val[1]));

									float32x4_t _r_0_m_6 = vsubq_f32(_r_00, _r_66);
									float32x4_t _r_7_m_1 = vsubq_f32(_r_77, _r_11);

									float32x4_t _r_4_m_2 = vsubq_f32(_r_44, _r_22);
									float32x4_t _r_3_m_5 = vsubq_f32(_r_33, _r_55);

									float32x4_t _tmp0 = vmlaq_lane_f32(_r_0_m_6, _r_4_m_2, vget_high_f32(_coeff1), 1);
									float32x4_t _tmp7 = vmlaq_lane_f32(_r_7_m_1, _r_3_m_5, vget_high_f32(_coeff1), 1);

									vst1q_f32(&tmp[0][m], _tmp0);
									vst1q_f32(&tmp[7][m], _tmp7);

									float32x4_t _r_2_a_6 = vaddq_f32(_r_22, _r_66);
									float32x4_t _r_1_a_5 = vaddq_f32(_r_11, _r_55);

									float32x4_t _tmp12a = vmlsq_lane_f32(_r_2_a_6, _r_44, vget_high_f32(_coeff1), 0);
									float32x4_t _tmp12b = vmlsq_lane_f32(_r_1_a_5, _r_33, vget_high_f32(_coeff1), 0);

									float32x4_t _tmp1 = vaddq_f32(_tmp12a, _tmp12b);
									float32x4_t _tmp2 = vsubq_f32(_tmp12a, _tmp12b);

									vst1q_f32(&tmp[1][m], _tmp1);
									vst1q_f32(&tmp[2][m], _tmp2);

									float32x4_t _r_4_x_c = vmulq_lane_f32(_r_44, vget_high_f32(_coeff0), 0);
									float32x4_t _r_3_x_c = vmulq_lane_f32(_r_33, vget_low_f32(_coeff1), 0);

									float32x4_t _tmp34a = vaddq_f32(_r_66, _r_4_x_c);
									_tmp34a = vmlaq_lane_f32(_tmp34a, _r_22, vget_low_f32(_coeff0), 0);

									float32x4_t _tmp34b = vmlaq_lane_f32(_r_3_x_c, _r_11, vget_low_f32(_coeff0), 1);
									_tmp34b = vmlaq_lane_f32(_tmp34b, _r_55, vget_high_f32(_coeff0), 1);

									float32x4_t _tmp3 = vaddq_f32(_tmp34a, _tmp34b);
									float32x4_t _tmp4 = vsubq_f32(_tmp34a, _tmp34b);

									vst1q_f32(&tmp[3][m], _tmp3);
									vst1q_f32(&tmp[4][m], _tmp4);

									// reuse r04 * 1.25
									// reuse r03 * 2.5
									float32x4_t _r_2_a_4c = vaddq_f32(_r_22, _r_4_x_c);
									float32x4_t _tmp56a = vmlaq_lane_f32(_r_66, _r_2_a_4c, vget_low_f32(_coeff1), 1);
									float32x4_t _tmp56b = vmlaq_lane_f32(_r_3_x_c, _r_11, vget_high_f32(_coeff0), 1);
									_tmp56b = vmlaq_lane_f32(_tmp56b, _r_55, vget_low_f32(_coeff0), 1);

									float32x4_t _tmp5 = vaddq_f32(_tmp56a, _tmp56b);
									float32x4_t _tmp6 = vsubq_f32(_tmp56a, _tmp56b);

									vst1q_f32(&tmp[5][m], _tmp5);
									vst1q_f32(&tmp[6][m], _tmp6);

									r0 += w * 4;
									r1 += w * 4;
									r2 += w * 4;
									r3 += w * 4;
								}

								const float* t0 = tmp[0];
								const float* t1 = tmp[1];
								const float* t2 = tmp[2];
								const float* t3 = tmp[3];

								float* r0_tm0 = img0_tm + (i * w_tm / 8 + j) * bottom_tm_w;
								float* r0_tm1 = img0_tm + (i * w_tm / 8 + j + tiles * 8) * bottom_tm_w;
								float* r0_tm2 = img0_tm + (i * w_tm / 8 + j + tiles * 16) * bottom_tm_w;
								float* r0_tm3 = img0_tm + (i * w_tm / 8 + j + tiles * 24) * bottom_tm_w;

								for (int m = 0; m + 3 < 8; m += 4)
								{
									float32x4_t _t0_0123 = vld1q_f32(t0);
									float32x4_t _t0_4567 = vld1q_f32(t0 + 4);
									float32x4_t _t1_0123 = vld1q_f32(t1);
									float32x4_t _t1_4567 = vld1q_f32(t1 + 4);
									float32x4_t _t2_0123 = vld1q_f32(t2);
									float32x4_t _t2_4567 = vld1q_f32(t2 + 4);
									float32x4_t _t3_0123 = vld1q_f32(t3);
									float32x4_t _t3_4567 = vld1q_f32(t3 + 4);

									float32x4x2_t _t01_00221133 = vtrnq_f32(_t0_0123, _t1_0123);
									float32x4x2_t _t01_44665577 = vtrnq_f32(_t0_4567, _t1_4567);
									float32x4x2_t _t23_00221133 = vtrnq_f32(_t2_0123, _t3_0123);
									float32x4x2_t _t23_44665577 = vtrnq_f32(_t2_4567, _t3_4567);

									// no vswp intrinsic  :(
									float32x4_t _t_00 = vcombine_f32(vget_low_f32(_t01_00221133.val[0]), vget_low_f32(_t23_00221133.val[0]));
									float32x4_t _t_11 = vcombine_f32(vget_low_f32(_t01_00221133.val[1]), vget_low_f32(_t23_00221133.val[1]));
									float32x4_t _t_22 = vcombine_f32(vget_high_f32(_t01_00221133.val[0]), vget_high_f32(_t23_00221133.val[0]));
									float32x4_t _t_33 = vcombine_f32(vget_high_f32(_t01_00221133.val[1]), vget_high_f32(_t23_00221133.val[1]));
									float32x4_t _t_44 = vcombine_f32(vget_low_f32(_t01_44665577.val[0]), vget_low_f32(_t23_44665577.val[0]));
									float32x4_t _t_55 = vcombine_f32(vget_low_f32(_t01_44665577.val[1]), vget_low_f32(_t23_44665577.val[1]));
									float32x4_t _t_66 = vcombine_f32(vget_high_f32(_t01_44665577.val[0]), vget_high_f32(_t23_44665577.val[0]));
									float32x4_t _t_77 = vcombine_f32(vget_high_f32(_t01_44665577.val[1]), vget_high_f32(_t23_44665577.val[1]));

									float32x4_t _t_0_m_6 = vsubq_f32(_t_00, _t_66);
									float32x4_t _t_7_m_1 = vsubq_f32(_t_77, _t_11);

									float32x4_t _t_4_m_2 = vsubq_f32(_t_44, _t_22);
									float32x4_t _t_3_m_5 = vsubq_f32(_t_33, _t_55);

									float32x4_t _r0_tm_0_0 = vmlaq_lane_f32(_t_0_m_6, _t_4_m_2, vget_high_f32(_coeff1), 1);
									float32x4_t _r0_tm_4_3 = vmlaq_lane_f32(_t_7_m_1, _t_3_m_5, vget_high_f32(_coeff1), 1);

									r0_tm0[0] = vgetq_lane_f32(_r0_tm_0_0, 0);
									r0_tm1[0] = vgetq_lane_f32(_r0_tm_0_0, 1);
									r0_tm2[0] = vgetq_lane_f32(_r0_tm_0_0, 2);
									r0_tm3[0] = vgetq_lane_f32(_r0_tm_0_0, 3);

									r0_tm0 += bottom_tm_w * tiles;
									r0_tm1 += bottom_tm_w * tiles;
									r0_tm2 += bottom_tm_w * tiles;
									r0_tm3 += bottom_tm_w * tiles;

									float32x4_t _t_2_m_6 = vaddq_f32(_t_22, _t_66);
									float32x4_t _t_1_m_5 = vaddq_f32(_t_11, _t_55);

									float32x4_t _tmp12a = vmlsq_lane_f32(_t_2_m_6, _t_44, vget_high_f32(_coeff1), 0);
									float32x4_t _tmp12b = vmlsq_lane_f32(_t_1_m_5, _t_33, vget_high_f32(_coeff1), 0);

									float32x4_t _r0_tm_0_1 = vaddq_f32(_tmp12a, _tmp12b);
									float32x4_t _r0_tm_0_2 = vsubq_f32(_tmp12a, _tmp12b);

									r0_tm0[0] = vgetq_lane_f32(_r0_tm_0_1, 0);
									r0_tm1[0] = vgetq_lane_f32(_r0_tm_0_1, 1);
									r0_tm2[0] = vgetq_lane_f32(_r0_tm_0_1, 2);
									r0_tm3[0] = vgetq_lane_f32(_r0_tm_0_1, 3);

									r0_tm0 += bottom_tm_w * tiles;
									r0_tm1 += bottom_tm_w * tiles;
									r0_tm2 += bottom_tm_w * tiles;
									r0_tm3 += bottom_tm_w * tiles;

									r0_tm0[0] = vgetq_lane_f32(_r0_tm_0_2, 0);
									r0_tm1[0] = vgetq_lane_f32(_r0_tm_0_2, 1);
									r0_tm2[0] = vgetq_lane_f32(_r0_tm_0_2, 2);
									r0_tm3[0] = vgetq_lane_f32(_r0_tm_0_2, 3);

									r0_tm0 += bottom_tm_w * tiles;
									r0_tm1 += bottom_tm_w * tiles;
									r0_tm2 += bottom_tm_w * tiles;
									r0_tm3 += bottom_tm_w * tiles;

									float32x4_t _t_4_x_c = vmulq_lane_f32(_t_44, vget_high_f32(_coeff0), 0);
									float32x4_t _t_3_x_c = vmulq_lane_f32(_t_33, vget_low_f32(_coeff1), 0);

									float32x4_t _tmp34a = vaddq_f32(_t_66, _t_4_x_c);
									_tmp34a = vmlaq_lane_f32(_tmp34a, _t_22, vget_low_f32(_coeff0), 0);

									float32x4_t _tmp34b = vmlaq_lane_f32(_t_3_x_c, _t_11, vget_low_f32(_coeff0), 1);
									_tmp34b = vmlaq_lane_f32(_tmp34b, _t_55, vget_high_f32(_coeff0), 1);

									float32x4_t _r0_tm_0_3 = vaddq_f32(_tmp34a, _tmp34b);
									float32x4_t _r0_tm_4_0 = vsubq_f32(_tmp34a, _tmp34b);

									r0_tm0[0] = vgetq_lane_f32(_r0_tm_0_3, 0);
									r0_tm1[0] = vgetq_lane_f32(_r0_tm_0_3, 1);
									r0_tm2[0] = vgetq_lane_f32(_r0_tm_0_3, 2);
									r0_tm3[0] = vgetq_lane_f32(_r0_tm_0_3, 3);

									r0_tm0 += bottom_tm_w * tiles;
									r0_tm1 += bottom_tm_w * tiles;
									r0_tm2 += bottom_tm_w * tiles;
									r0_tm3 += bottom_tm_w * tiles;

									r0_tm0[0] = vgetq_lane_f32(_r0_tm_4_0, 0);
									r0_tm1[0] = vgetq_lane_f32(_r0_tm_4_0, 1);
									r0_tm2[0] = vgetq_lane_f32(_r0_tm_4_0, 2);
									r0_tm3[0] = vgetq_lane_f32(_r0_tm_4_0, 3);

									r0_tm0 += bottom_tm_w * tiles;
									r0_tm1 += bottom_tm_w * tiles;
									r0_tm2 += bottom_tm_w * tiles;
									r0_tm3 += bottom_tm_w * tiles;

									float32x4_t _t_2_a_4c = vaddq_f32(_t_22, _t_4_x_c);
									float32x4_t _tmp56a = vmlaq_lane_f32(_t_66, _t_2_a_4c, vget_low_f32(_coeff1), 1);
									float32x4_t _tmp56b = vmlaq_lane_f32(_t_3_x_c, _t_11, vget_high_f32(_coeff0), 1);
									_tmp56b = vmlaq_lane_f32(_tmp56b, _t_55, vget_low_f32(_coeff0), 1);

									float32x4_t _r0_tm_4_1 = vaddq_f32(_tmp56a, _tmp56b);
									float32x4_t _r0_tm_4_2 = vsubq_f32(_tmp56a, _tmp56b);

									r0_tm0[0] = vgetq_lane_f32(_r0_tm_4_1, 0);
									r0_tm1[0] = vgetq_lane_f32(_r0_tm_4_1, 1);
									r0_tm2[0] = vgetq_lane_f32(_r0_tm_4_1, 2);
									r0_tm3[0] = vgetq_lane_f32(_r0_tm_4_1, 3);

									r0_tm0 += bottom_tm_w * tiles;
									r0_tm1 += bottom_tm_w * tiles;
									r0_tm2 += bottom_tm_w * tiles;
									r0_tm3 += bottom_tm_w * tiles;

									r0_tm0[0] = vgetq_lane_f32(_r0_tm_4_2, 0);
									r0_tm1[0] = vgetq_lane_f32(_r0_tm_4_2, 1);
									r0_tm2[0] = vgetq_lane_f32(_r0_tm_4_2, 2);
									r0_tm3[0] = vgetq_lane_f32(_r0_tm_4_2, 3);

									r0_tm0 += bottom_tm_w * tiles;
									r0_tm1 += bottom_tm_w * tiles;
									r0_tm2 += bottom_tm_w * tiles;
									r0_tm3 += bottom_tm_w * tiles;

									r0_tm0[0] = vgetq_lane_f32(_r0_tm_4_3, 0);
									r0_tm1[0] = vgetq_lane_f32(_r0_tm_4_3, 1);
									r0_tm2[0] = vgetq_lane_f32(_r0_tm_4_3, 2);
									r0_tm3[0] = vgetq_lane_f32(_r0_tm_4_3, 3);

									t0 += 8 * 4;
									t1 += 8 * 4;
									t2 += 8 * 4;
									t3 += 8 * 4;

									r0_tm0 += bottom_tm_w * tiles * 25;
									r0_tm1 += bottom_tm_w * tiles * 25;
									r0_tm2 += bottom_tm_w * tiles * 25;
									r0_tm3 += bottom_tm_w * tiles * 25;
								}
#else // __aarch64__
								float* t0 = tmp[0];
								float* t1 = tmp[1];
								float* t2 = tmp[2];
								float* t3 = tmp[3];
								float* t4 = tmp[4];
								float* t5 = tmp[5];
								float* t6 = tmp[6];
								float* t7 = tmp[7];

								int stepw = w * 4 * 4;

								asm volatile(

									// loop0
									"vld1.f32   {d16-d19}, [%8], %26    \n"
									"vld1.f32   {d20-d23}, [%9], %26    \n"
									"vld1.f32   {d24-d27}, [%10], %26   \n"

									"vtrn.32    q8, q10             \n"

									"vld1.f32   {d28-d31}, [%11], %26   \n"

									"vtrn.32    q9, q11             \n"
									"vtrn.32    q12, q14            \n"
									"vtrn.32    q13, q15            \n"

									"vswp       d17, d24            \n"
									"vswp       d19, d26            \n"
									"vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
									"vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77

									"vsub.f32   q2, q8, q13         \n"
									"vsub.f32   q3, q9, q12         \n"

									"vadd.f32   q4, q12, q13        \n"
									"vadd.f32   q5, q10, q11        \n"

									"vmla.f32   q2, q3, %f25[1]     \n"

									"vmul.f32   q7, q14, %e25[0]    \n"// q7 = _r_3_x_c
									"vmul.f32   q6, q9, %f24[0]     \n"// q6 = _r_4_x_c

									"vmls.f32   q4, q9, %f25[0]     \n"
									"vmls.f32   q5, q14, %f25[0]    \n"

									"vst1.f32   {d4-d5}, [%0]!      \n"// tmp[0][m]

									"vmov       q3, q7              \n"// use q7

									"vadd.f32   q2, q13, q6         \n"// use q6
									"vmla.f32   q3, q10, %e24[1]    \n"

									"vadd.f32   q8, q4, q5          \n"
									"vsub.f32   q9, q4, q5          \n"

									"vmov       q5, q7              \n"// use q7

									"vadd.f32   q6, q12, q6         \n"// use q6
									"vmla.f32   q5, q10, %f24[1]    \n"

									"vmov       q4, q13             \n"

									"vmla.f32   q2, q12, %e24[0]    \n"
									"vmla.f32   q3, q11, %f24[1]    \n"

									"vst1.f32   {d16-d17}, [%1]!    \n"// tmp[1][m]

									"vmla.f32   q4, q6, %e25[1]     \n"
									"vmla.f32   q5, q11, %e24[1]    \n"

									"vst1.f32   {d18-d19}, [%2]!    \n"// tmp[2][m]

									"vadd.f32   q8, q2, q3          \n"
									"vsub.f32   q9, q2, q3          \n"

									"vsub.f32   q6, q15, q10        \n"
									"vsub.f32   q7, q14, q11        \n"

									"vadd.f32   q2, q4, q5          \n"
									"vsub.f32   q3, q4, q5          \n"

									"vst1.f32   {d16-d17}, [%3]!    \n"// tmp[3][m]
									"vst1.f32   {d18-d19}, [%4]!    \n"// tmp[4][m]

									"vmla.f32   q6, q7, %f25[1]     \n"

									"vst1.f32   {d4-d5}, [%5]!      \n"// tmp[5][m]
									"vst1.f32   {d6-d7}, [%6]!      \n"// tmp[6][m]

									"vst1.f32   {d12-d13}, [%7]!    \n"// tmp[7][m]

																	   // loop1
									"vld1.f32   {d16-d19}, [%8]     \n"
									"vld1.f32   {d20-d23}, [%9]     \n"
									"vld1.f32   {d24-d27}, [%10]    \n"

									"vtrn.32    q8, q10             \n"

									"vld1.f32   {d28-d31}, [%11]    \n"

									"vtrn.32    q9, q11             \n"
									"vtrn.32    q12, q14            \n"
									"vtrn.32    q13, q15            \n"

									"vswp       d17, d24            \n"
									"vswp       d19, d26            \n"
									"vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
									"vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77

									"vsub.f32   q2, q8, q13         \n"
									"vsub.f32   q3, q9, q12         \n"

									"vadd.f32   q4, q12, q13        \n"
									"vadd.f32   q5, q10, q11        \n"

									"vmla.f32   q2, q3, %f25[1]     \n"

									"vmul.f32   q7, q14, %e25[0]    \n"// q7 = _r_3_x_c
									"vmul.f32   q6, q9, %f24[0]     \n"// q6 = _r_4_x_c

									"vmls.f32   q4, q9, %f25[0]     \n"
									"vmls.f32   q5, q14, %f25[0]    \n"

									"vst1.f32   {d4-d5}, [%0]!      \n"// tmp[0][m]

									"vmov       q3, q7              \n"// use q7

									"vadd.f32   q2, q13, q6         \n"// use q6
									"vmla.f32   q3, q10, %e24[1]    \n"

									"vadd.f32   q8, q4, q5          \n"
									"vsub.f32   q9, q4, q5          \n"

									"vmov       q5, q7              \n"// use q7

									"vadd.f32   q6, q12, q6         \n"// use q6
									"vmla.f32   q5, q10, %f24[1]    \n"

									"vmov       q4, q13             \n"

									"vmla.f32   q2, q12, %e24[0]    \n"
									"vmla.f32   q3, q11, %f24[1]    \n"

									"vst1.f32   {d16-d17}, [%1]!    \n"// tmp[1][m]

									"vmla.f32   q4, q6, %e25[1]     \n"
									"vmla.f32   q5, q11, %e24[1]    \n"

									"vst1.f32   {d18-d19}, [%2]!    \n"// tmp[2][m]

									"vadd.f32   q8, q2, q3          \n"
									"vsub.f32   q9, q2, q3          \n"

									"vsub.f32   q6, q15, q10        \n"
									"vsub.f32   q7, q14, q11        \n"

									"vadd.f32   q2, q4, q5          \n"
									"vsub.f32   q3, q4, q5          \n"

									"vst1.f32   {d16-d17}, [%3]!    \n"// tmp[3][m]
									"vst1.f32   {d18-d19}, [%4]!    \n"// tmp[4][m]

									"vmla.f32   q6, q7, %f25[1]     \n"

									"vst1.f32   {d4-d5}, [%5]!      \n"// tmp[5][m]
									"vst1.f32   {d6-d7}, [%6]!      \n"// tmp[6][m]

									"vst1.f32   {d12-d13}, [%7]!    \n"// tmp[7][m]

									: "=r"(t0),     // %0
									"=r"(t1),     // %1
									"=r"(t2),     // %2
									"=r"(t3),     // %3
									"=r"(t4),     // %4
									"=r"(t5),     // %5
									"=r"(t6),     // %6
									"=r"(t7),     // %7
									"=r"(r0),     // %8
									"=r"(r1),     // %9
									"=r"(r2),     // %10
									"=r"(r3)      // %11
									: "0"(t0),
									"1"(t1),
									"2"(t2),
									"3"(t3),
									"4"(t4),
									"5"(t5),
									"6"(t6),
									"7"(t7),
									"8"(r0),
									"9"(r1),
									"10"(r2),
									"11"(r3),
									"w"(_coeff0), // %24
									"w"(_coeff1), // %25
									"r"(stepw)        // %26
									: "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
									);

								t0 = tmp[0];
								t1 = tmp[1];
								t2 = tmp[2];
								t3 = tmp[3];

								float* r0_tm0_0 = img0_tm + (i * w_tm / 8 + j) * bottom_tm_w;
								float* r0_tm1_0 = img0_tm + (i * w_tm / 8 + j + tiles * 8) * bottom_tm_w;
								float* r0_tm2_0 = img0_tm + (i * w_tm / 8 + j + tiles * 16) * bottom_tm_w;
								float* r0_tm3_0 = img0_tm + (i * w_tm / 8 + j + tiles * 24) * bottom_tm_w;
								float* r0_tm0_4 = img0_tm + (i * w_tm / 8 + j + tiles * 32) * bottom_tm_w;
								float* r0_tm1_4 = img0_tm + (i * w_tm / 8 + j + tiles * 40) * bottom_tm_w;
								float* r0_tm2_4 = img0_tm + (i * w_tm / 8 + j + tiles * 48) * bottom_tm_w;
								float* r0_tm3_4 = img0_tm + (i * w_tm / 8 + j + tiles * 56) * bottom_tm_w;

								int step = bottom_tm_w * tiles * 4;

								asm volatile(

									// loop0
									"vld1.f32   {d16-d19}, [%8]     \n"
									"add        %8, %8, #128        \n"
									"vld1.f32   {d20-d23}, [%9]     \n"
									"add        %9, %9, #128        \n"
									"vld1.f32   {d24-d27}, [%10]    \n"
									"add        %10, %10, #128      \n"

									"vtrn.32    q8, q10             \n"

									"vld1.f32   {d28-d31}, [%11]    \n"
									"add        %11, %11, #128      \n"

									"vtrn.32    q9, q11             \n"
									"vtrn.32    q12, q14            \n"
									"vtrn.32    q13, q15            \n"

									"vswp       d17, d24            \n"
									"vswp       d19, d26            \n"
									"vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
									"vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77

									"vsub.f32   q2, q8, q13         \n"
									"vsub.f32   q3, q9, q12         \n"

									"vadd.f32   q4, q12, q13        \n"
									"vadd.f32   q5, q10, q11        \n"

									"vmla.f32   q2, q3, %f25[1]     \n"

									"vmul.f32   q7, q14, %e25[0]    \n"// q7 = _r_3_x_c
									"vmul.f32   q6, q9, %f24[0]     \n"// q6 = _r_4_x_c

									"vmls.f32   q4, q9, %f25[0]     \n"
									"vmls.f32   q5, q14, %f25[0]    \n"

									"vst1.f32   {d4[0]}, [%0], %26  \n"
									"vst1.f32   {d4[1]}, [%1], %26  \n"

									"vmov       q3, q7              \n"// use q7

									"vst1.f32   {d5[0]}, [%2], %26  \n"
									"vst1.f32   {d5[1]}, [%3], %26  \n"

									"vadd.f32   q2, q13, q6         \n"// use q6
									"vmla.f32   q3, q10, %e24[1]    \n"

									"vadd.f32   q8, q4, q5          \n"
									"vsub.f32   q9, q4, q5          \n"

									"vmov       q5, q7              \n"// use q7

									"vadd.f32   q6, q12, q6         \n"// use q6
									"vmla.f32   q5, q10, %f24[1]    \n"

									"vmov       q4, q13             \n"

									"vmla.f32   q2, q12, %e24[0]    \n"
									"vmla.f32   q3, q11, %f24[1]    \n"

									"vst1.f32   {d16[0]}, [%0], %26 \n"
									"vst1.f32   {d16[1]}, [%1], %26 \n"

									"vmla.f32   q4, q6, %e25[1]     \n"

									"vst1.f32   {d17[0]}, [%2], %26 \n"
									"vst1.f32   {d17[1]}, [%3], %26 \n"

									"vmla.f32   q5, q11, %e24[1]    \n"

									"vst1.f32   {d18[0]}, [%0], %26 \n"
									"vst1.f32   {d18[1]}, [%1], %26 \n"

									"vadd.f32   q8, q2, q3          \n"

									"vst1.f32   {d19[0]}, [%2], %26 \n"
									"vst1.f32   {d19[1]}, [%3], %26 \n"

									"vsub.f32   q9, q2, q3          \n"

									"vsub.f32   q6, q15, q10        \n"
									"vsub.f32   q7, q14, q11        \n"

									"vst1.f32   {d16[0]}, [%0], %26 \n"
									"vst1.f32   {d16[1]}, [%1], %26 \n"
									"vst1.f32   {d17[0]}, [%2], %26 \n"
									"vst1.f32   {d17[1]}, [%3], %26 \n"

									"vadd.f32   q2, q4, q5          \n"

									"vst1.f32   {d18[0]}, [%0], %26 \n"
									"vst1.f32   {d18[1]}, [%1], %26 \n"
									"vst1.f32   {d19[0]}, [%2], %26 \n"
									"vst1.f32   {d19[1]}, [%3], %26 \n"

									"vsub.f32   q3, q4, q5          \n"

									"vst1.f32   {d4[0]}, [%0], %26  \n"
									"vst1.f32   {d4[1]}, [%1], %26  \n"
									"vst1.f32   {d5[0]}, [%2], %26  \n"
									"vst1.f32   {d5[1]}, [%3], %26  \n"

									"vmla.f32   q6, q7, %f25[1]     \n"

									"vst1.f32   {d6[0]}, [%0], %26  \n"
									"vst1.f32   {d6[1]}, [%1], %26  \n"
									"vst1.f32   {d7[0]}, [%2], %26  \n"
									"vst1.f32   {d7[1]}, [%3], %26  \n"

									"vst1.f32   {d12[0]}, [%0]      \n"
									"vst1.f32   {d12[1]}, [%1]      \n"
									"vst1.f32   {d13[0]}, [%2]      \n"
									"vst1.f32   {d13[1]}, [%3]      \n"

									// loop1
									"vld1.f32   {d16-d19}, [%8]     \n"
									"vld1.f32   {d20-d23}, [%9]     \n"
									"vld1.f32   {d24-d27}, [%10]    \n"

									"vtrn.32    q8, q10             \n"

									"vld1.f32   {d28-d31}, [%11]    \n"

									"vtrn.32    q9, q11             \n"
									"vtrn.32    q12, q14            \n"
									"vtrn.32    q13, q15            \n"

									"vswp       d17, d24            \n"
									"vswp       d19, d26            \n"
									"vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
									"vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77

									"vsub.f32   q2, q8, q13         \n"
									"vsub.f32   q3, q9, q12         \n"

									"vadd.f32   q4, q12, q13        \n"
									"vadd.f32   q5, q10, q11        \n"

									"vmla.f32   q2, q3, %f25[1]     \n"

									"vmul.f32   q7, q14, %e25[0]    \n"// q7 = _r_3_x_c
									"vmul.f32   q6, q9, %f24[0]     \n"// q6 = _r_4_x_c

									"vmls.f32   q4, q9, %f25[0]     \n"
									"vmls.f32   q5, q14, %f25[0]    \n"

									"vst1.f32   {d4[0]}, [%4], %26  \n"
									"vst1.f32   {d4[1]}, [%5], %26  \n"

									"vmov       q3, q7              \n"// use q7

									"vst1.f32   {d5[0]}, [%6], %26  \n"
									"vst1.f32   {d5[1]}, [%7], %26  \n"

									"vadd.f32   q2, q13, q6         \n"// use q6
									"vmla.f32   q3, q10, %e24[1]    \n"

									"vadd.f32   q8, q4, q5          \n"
									"vsub.f32   q9, q4, q5          \n"

									"vmov       q5, q7              \n"// use q7

									"vadd.f32   q6, q12, q6         \n"// use q6
									"vmla.f32   q5, q10, %f24[1]    \n"

									"vmov       q4, q13             \n"

									"vmla.f32   q2, q12, %e24[0]    \n"
									"vmla.f32   q3, q11, %f24[1]    \n"

									"vst1.f32   {d16[0]}, [%4], %26 \n"
									"vst1.f32   {d16[1]}, [%5], %26 \n"

									"vmla.f32   q4, q6, %e25[1]     \n"

									"vst1.f32   {d17[0]}, [%6], %26 \n"
									"vst1.f32   {d17[1]}, [%7], %26 \n"

									"vmla.f32   q5, q11, %e24[1]    \n"

									"vst1.f32   {d18[0]}, [%4], %26 \n"
									"vst1.f32   {d18[1]}, [%5], %26 \n"

									"vadd.f32   q8, q2, q3          \n"

									"vst1.f32   {d19[0]}, [%6], %26 \n"
									"vst1.f32   {d19[1]}, [%7], %26 \n"

									"vsub.f32   q9, q2, q3          \n"

									"vsub.f32   q6, q15, q10        \n"
									"vsub.f32   q7, q14, q11        \n"

									"vst1.f32   {d16[0]}, [%4], %26 \n"
									"vst1.f32   {d16[1]}, [%5], %26 \n"
									"vst1.f32   {d17[0]}, [%6], %26 \n"
									"vst1.f32   {d17[1]}, [%7], %26 \n"

									"vadd.f32   q2, q4, q5          \n"

									"vst1.f32   {d18[0]}, [%4], %26 \n"
									"vst1.f32   {d18[1]}, [%5], %26 \n"
									"vst1.f32   {d19[0]}, [%6], %26 \n"
									"vst1.f32   {d19[1]}, [%7], %26 \n"

									"vsub.f32   q3, q4, q5          \n"

									"vst1.f32   {d4[0]}, [%4], %26  \n"
									"vst1.f32   {d4[1]}, [%5], %26  \n"
									"vst1.f32   {d5[0]}, [%6], %26  \n"
									"vst1.f32   {d5[1]}, [%7], %26  \n"

									"vmla.f32   q6, q7, %f25[1]     \n"

									"vst1.f32   {d6[0]}, [%4], %26  \n"
									"vst1.f32   {d6[1]}, [%5], %26  \n"
									"vst1.f32   {d7[0]}, [%6], %26  \n"
									"vst1.f32   {d7[1]}, [%7], %26  \n"

									"vst1.f32   {d12[0]}, [%4]      \n"
									"vst1.f32   {d12[1]}, [%5]      \n"
									"vst1.f32   {d13[0]}, [%6]      \n"
									"vst1.f32   {d13[1]}, [%7]      \n"

									: "=r"(r0_tm0_0),     // %0
									"=r"(r0_tm1_0),     // %1
									"=r"(r0_tm2_0),     // %2
									"=r"(r0_tm3_0),     // %3
									"=r"(r0_tm0_4),     // %4
									"=r"(r0_tm1_4),     // %5
									"=r"(r0_tm2_4),     // %6
									"=r"(r0_tm3_4),     // %7
									"=r"(t0),     // %8
									"=r"(t1),     // %9
									"=r"(t2),     // %10
									"=r"(t3)      // %11
									: "0"(r0_tm0_0),
									"1"(r0_tm1_0),
									"2"(r0_tm2_0),
									"3"(r0_tm3_0),
									"4"(r0_tm0_4),
									"5"(r0_tm1_4),
									"6"(r0_tm2_4),
									"7"(r0_tm3_4),
									"8"(t0),
									"9"(t1),
									"10"(t2),
									"11"(t3),
									"w"(_coeff0), // %24
									"w"(_coeff1), // %25
									"r"(step)        // %26
									: "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
									);
#endif // __aarch64__
#else
								const float* r0 = img0 + (i * 6) * w + j * 6;

								for (int m = 0; m < 8; m++)
								{
									tmp[0][m] = r0[0] - r0[6] + (r0[4] - r0[2]) * 5.25f;
									tmp[7][m] = r0[7] - r0[1] + (r0[3] - r0[5]) * 5.25f;

									float tmp12a = (r0[2] + r0[6] - r0[4] * 4.25f);
									float tmp12b = (r0[1] + r0[5] - r0[3] * 4.25f);

									tmp[1][m] = tmp12a + tmp12b;
									tmp[2][m] = tmp12a - tmp12b;

									float tmp34a = (r0[6] + r0[2] * 0.25f - r0[4] * 1.25f);
									float tmp34b = (r0[1] * 0.5f - r0[3] * 2.5f + r0[5] * 2.f);

									tmp[3][m] = tmp34a + tmp34b;
									tmp[4][m] = tmp34a - tmp34b;

									float tmp56a = (r0[6] + (r0[2] - r0[4] * 1.25f) * 4.f);
									float tmp56b = (r0[1] * 2.f - r0[3] * 2.5f + r0[5] * 0.5f);

									tmp[5][m] = tmp56a + tmp56b;
									tmp[6][m] = tmp56a - tmp56b;

									r0 += w;
								}

								float* r0_tm_0 = img0_tm + (i * w_tm / 8 + j) * bottom_tm_w;
								float* r0_tm_1 = img0_tm + (i * w_tm / 8 + j + tiles) * bottom_tm_w;
								float* r0_tm_2 = img0_tm + (i * w_tm / 8 + j + tiles * 2) * bottom_tm_w;
								float* r0_tm_3 = img0_tm + (i * w_tm / 8 + j + tiles * 3) * bottom_tm_w;
								float* r0_tm_4 = img0_tm + (i * w_tm / 8 + j + tiles * 4) * bottom_tm_w;
								float* r0_tm_5 = img0_tm + (i * w_tm / 8 + j + tiles * 5) * bottom_tm_w;
								float* r0_tm_6 = img0_tm + (i * w_tm / 8 + j + tiles * 6) * bottom_tm_w;
								float* r0_tm_7 = img0_tm + (i * w_tm / 8 + j + tiles * 7) * bottom_tm_w;

								for (int m = 0; m < 8; m++)
								{
									const float* tmp0 = tmp[m];

									r0_tm_0[0] = tmp0[0] - tmp0[6] + (tmp0[4] - tmp0[2]) * 5.25f;
									r0_tm_7[0] = tmp0[7] - tmp0[1] + (tmp0[3] - tmp0[5]) * 5.25f;

									float tmp12a = (tmp0[2] + tmp0[6] - tmp0[4] * 4.25f);
									float tmp12b = (tmp0[1] - tmp0[3] * 4.25f + tmp0[5]);

									r0_tm_1[0] = tmp12a + tmp12b;
									r0_tm_2[0] = tmp12a - tmp12b;

									float tmp34a = (tmp0[6] + tmp0[2] * 0.25f - tmp0[4] * 1.25f);
									float tmp34b = (tmp0[1] * 0.5f - tmp0[3] * 2.5f + tmp0[5] * 2.f);

									r0_tm_3[0] = tmp34a + tmp34b;
									r0_tm_4[0] = tmp34a - tmp34b;

									float tmp56a = (tmp0[6] + (tmp0[2] - tmp0[4] * 1.25f) * 4.f);
									float tmp56b = (tmp0[1] * 2.f - tmp0[3] * 2.5f + tmp0[5] * 0.5f);

									r0_tm_5[0] = tmp56a + tmp56b;
									r0_tm_6[0] = tmp56a - tmp56b;

									r0_tm_0 += bottom_tm_w * tiles * 8;
									r0_tm_1 += bottom_tm_w * tiles * 8;
									r0_tm_2 += bottom_tm_w * tiles * 8;
									r0_tm_3 += bottom_tm_w * tiles * 8;
									r0_tm_4 += bottom_tm_w * tiles * 8;
									r0_tm_5 += bottom_tm_w * tiles * 8;
									r0_tm_6 += bottom_tm_w * tiles * 8;
									r0_tm_7 += bottom_tm_w * tiles * 8;
								}
#endif // __ARM_NEON
							}
						}
					}
				}
				// END transform input

				// BEGIN dot

				{
					// permute
#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
					for (int r = 0; r < 64; r++)
					{
						float *tm2 = bottom_tm2_data + (r)* bottom_tm2_cstep;

						// tile
						int i = 0;
						for (; i + 7 < tiles; i += 8)
						{
							float* tm2p = tm2 + (i / 8) * bottom_tm2_w;

							const float* r0 = bottom_tm_data;

							r0 += r * tiles + i;

							for (int q = 0; q < inch; q++)
							{
#if __ARM_NEON
								float32x4_t _r0 = vld1q_f32(r0);
								float32x4_t _r0n = vld1q_f32(r0 + 4);
								vst1q_f32(tm2p, _r0);
								vst1q_f32(tm2p + 4, _r0n);
#else
								tm2p[0] = r0[0];
								tm2p[1] = r0[1];
								tm2p[2] = r0[2];
								tm2p[3] = r0[3];
								tm2p[4] = r0[4];
								tm2p[5] = r0[5];
								tm2p[6] = r0[6];
								tm2p[7] = r0[7];
#endif // __ARM_NEON

								r0 += bottom_tm_cstep;
								tm2p += 8;
							}
						}
						for (; i + 3 < tiles; i += 4)
						{
							float* tm2p = tm2 + (i / 8 + (i % 8) / 4) * bottom_tm2_w;

							const float* r0 = bottom_tm_data;

							r0 += r * tiles + i;

							for (int q = 0; q < inch; q++)
							{
#if __ARM_NEON
								float32x4_t _r0 = vld1q_f32(r0);
								vst1q_f32(tm2p, _r0);
#else
								tm2p[0] = r0[0];
								tm2p[1] = r0[1];
								tm2p[2] = r0[2];
								tm2p[3] = r0[3];
#endif // __ARM_NEON

								r0 += bottom_tm_cstep;
								tm2p += 4;
							}
						}
						for (; i < tiles; i++)
						{
							float* tm2p = tm2 + (i / 8 + (i % 8) / 4 + i % 4) * bottom_tm2_w;

							const float* r0 = bottom_tm_data;

							r0 += r * tiles + i;

							for (int q = 0; q < inch; q++)
							{
								tm2p[0] = r0[0];

								r0 += bottom_tm_cstep;
								tm2p += 1;
							}
						}
					}
					// permute end

					int nn_outch = 0;
					int remain_outch_start = 0;

#if __ARM_NEON && __aarch64__
					nn_outch = outch >> 3;
					remain_outch_start = nn_outch << 3;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
					for (int pp = 0; pp < nn_outch; pp++)
					{
						int p = pp * 8;

						const float *kernel_tm0 = kernel_tm_data + (p / 8) * kernel_tm_cstep;

						float *out0_tm = top_tm_data + (p + 0) * top_tm_cstep;
						float *out1_tm = top_tm_data + (p + 1) * top_tm_cstep;
						float *out2_tm = top_tm_data + (p + 2) * top_tm_cstep;
						float *out3_tm = top_tm_data + (p + 3) * top_tm_cstep;
						float *out4_tm = top_tm_data + (p + 4) * top_tm_cstep;
						float *out5_tm = top_tm_data + (p + 5) * top_tm_cstep;
						float *out6_tm = top_tm_data + (p + 6) * top_tm_cstep;
						float *out7_tm = top_tm_data + (p + 7) * top_tm_cstep;

						float* output0_tm = out0_tm;
						float* output1_tm = out1_tm;
						float* output2_tm = out2_tm;
						float* output3_tm = out3_tm;
						float* output4_tm = out4_tm;
						float* output5_tm = out5_tm;
						float* output6_tm = out6_tm;
						float* output7_tm = out7_tm;

						for (int r = 0; r < 64; r++)
						{
							const float *bb2 = bottom_tm2_data + (r)* bottom_tm2_cstep;

							// tile
							int i = 0;
							for (; i + 7 < tiles; i += 8)
							{
								const float* bb2p0 = bb2 + (i / 8) * bottom_tm2_w;

								const float* ktm0 = kernel_tm0 + (r)* kernel_tm_w;

								asm volatile(
									"eor    v16.16b, v16.16b, v16.16b  \n"
									"eor    v17.16b, v17.16b, v17.16b  \n"
									"eor    v18.16b, v18.16b, v18.16b  \n"
									"eor    v19.16b, v19.16b, v19.16b  \n"
									"eor    v20.16b, v20.16b, v20.16b  \n"
									"eor    v21.16b, v21.16b, v21.16b  \n"
									"eor    v22.16b, v22.16b, v22.16b  \n"
									"eor    v23.16b, v23.16b, v23.16b  \n"
									"eor    v24.16b, v24.16b, v24.16b  \n"
									"eor    v25.16b, v25.16b, v25.16b  \n"
									"eor    v26.16b, v26.16b, v26.16b  \n"
									"eor    v27.16b, v27.16b, v27.16b  \n"
									"eor    v28.16b, v28.16b, v28.16b  \n"
									"eor    v29.16b, v29.16b, v29.16b  \n"
									"eor    v30.16b, v30.16b, v30.16b  \n"
									"eor    v31.16b, v31.16b, v31.16b  \n"

									// inch loop
									"lsr    w4, %w20, #2            \n"// w4 = nn = inch >> 2
									"cmp    w4, #0                  \n"
									"beq    1f                      \n"

									"0:                             \n"

									"prfm   pldl1keep, [%8, #512]   \n"
									"ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%8], #64   \n"

									"prfm   pldl1keep, [%9, #512]   \n"
									"ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64   \n"

									"fmla   v16.4s, v8.4s, v0.s[0]  \n"
									"fmla   v17.4s, v9.4s, v0.s[0]  \n"
									"fmla   v18.4s, v8.4s, v0.s[1]  \n"
									"fmla   v19.4s, v9.4s, v0.s[1]  \n"
									"fmla   v20.4s, v8.4s, v0.s[2]  \n"
									"fmla   v21.4s, v9.4s, v0.s[2]  \n"
									"fmla   v22.4s, v8.4s, v0.s[3]  \n"
									"fmla   v23.4s, v9.4s, v0.s[3]  \n"

									"prfm   pldl1keep, [%9, #512]   \n"
									"ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%9], #64   \n"

									"fmla   v24.4s, v8.4s, v1.s[0]  \n"
									"fmla   v25.4s, v9.4s, v1.s[0]  \n"
									"fmla   v26.4s, v8.4s, v1.s[1]  \n"
									"fmla   v27.4s, v9.4s, v1.s[1]  \n"
									"fmla   v28.4s, v8.4s, v1.s[2]  \n"
									"fmla   v29.4s, v9.4s, v1.s[2]  \n"
									"fmla   v30.4s, v8.4s, v1.s[3]  \n"
									"fmla   v31.4s, v9.4s, v1.s[3]  \n"

									"fmla   v16.4s, v10.4s, v2.s[0] \n"
									"fmla   v17.4s, v11.4s, v2.s[0] \n"
									"fmla   v18.4s, v10.4s, v2.s[1] \n"
									"fmla   v19.4s, v11.4s, v2.s[1] \n"
									"fmla   v20.4s, v10.4s, v2.s[2] \n"
									"fmla   v21.4s, v11.4s, v2.s[2] \n"
									"fmla   v22.4s, v10.4s, v2.s[3] \n"
									"fmla   v23.4s, v11.4s, v2.s[3] \n"

									"prfm   pldl1keep, [%8, #512]   \n"
									"ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%8], #64 \n"

									"fmla   v24.4s, v10.4s, v3.s[0] \n"
									"fmla   v25.4s, v11.4s, v3.s[0] \n"
									"fmla   v26.4s, v10.4s, v3.s[1] \n"
									"fmla   v27.4s, v11.4s, v3.s[1] \n"
									"fmla   v28.4s, v10.4s, v3.s[2] \n"
									"fmla   v29.4s, v11.4s, v3.s[2] \n"
									"fmla   v30.4s, v10.4s, v3.s[3] \n"
									"fmla   v31.4s, v11.4s, v3.s[3] \n"

									"fmla   v16.4s, v12.4s, v4.s[0] \n"
									"fmla   v17.4s, v13.4s, v4.s[0] \n"
									"fmla   v18.4s, v12.4s, v4.s[1] \n"
									"fmla   v19.4s, v13.4s, v4.s[1] \n"
									"fmla   v20.4s, v12.4s, v4.s[2] \n"
									"fmla   v21.4s, v13.4s, v4.s[2] \n"
									"fmla   v22.4s, v12.4s, v4.s[3] \n"
									"fmla   v23.4s, v13.4s, v4.s[3] \n"

									"fmla   v24.4s, v12.4s, v5.s[0] \n"
									"fmla   v25.4s, v13.4s, v5.s[0] \n"
									"fmla   v26.4s, v12.4s, v5.s[1] \n"
									"fmla   v27.4s, v13.4s, v5.s[1] \n"
									"fmla   v28.4s, v12.4s, v5.s[2] \n"
									"fmla   v29.4s, v13.4s, v5.s[2] \n"
									"fmla   v30.4s, v12.4s, v5.s[3] \n"
									"fmla   v31.4s, v13.4s, v5.s[3] \n"

									"fmla   v16.4s, v14.4s, v6.s[0] \n"
									"fmla   v17.4s, v15.4s, v6.s[0] \n"
									"fmla   v18.4s, v14.4s, v6.s[1] \n"
									"fmla   v19.4s, v15.4s, v6.s[1] \n"
									"fmla   v20.4s, v14.4s, v6.s[2] \n"
									"fmla   v21.4s, v15.4s, v6.s[2] \n"
									"fmla   v22.4s, v14.4s, v6.s[3] \n"
									"fmla   v23.4s, v15.4s, v6.s[3] \n"

									"subs   w4, w4, #1              \n"

									"fmla   v24.4s, v14.4s, v7.s[0] \n"
									"fmla   v25.4s, v15.4s, v7.s[0] \n"
									"fmla   v26.4s, v14.4s, v7.s[1] \n"
									"fmla   v27.4s, v15.4s, v7.s[1] \n"
									"fmla   v28.4s, v14.4s, v7.s[2] \n"
									"fmla   v29.4s, v15.4s, v7.s[2] \n"
									"fmla   v30.4s, v14.4s, v7.s[3] \n"
									"fmla   v31.4s, v15.4s, v7.s[3] \n"

									"bne    0b                      \n"

									"1:                             \n"

									// remain loop
									"and    w4, %w20, #3            \n"// w4 = remain = tiles & 3;
									"cmp    w4, #0                  \n"
									"beq    3f                      \n"

									"2:                             \n"

									"prfm   pldl1keep, [%8, #256]   \n"
									"ld1    {v8.4s, v9.4s}, [%8], #32   \n"

									"prfm   pldl1keep, [%9, #256]   \n"
									"ld1    {v0.4s, v1.4s}, [%9], #32   \n"

									"fmla   v16.4s, v8.4s, v0.s[0]  \n"
									"fmla   v17.4s, v9.4s, v0.s[0]  \n"
									"fmla   v18.4s, v8.4s, v0.s[1]  \n"
									"fmla   v19.4s, v9.4s, v0.s[1]  \n"
									"fmla   v20.4s, v8.4s, v0.s[2]  \n"
									"fmla   v21.4s, v9.4s, v0.s[2]  \n"
									"fmla   v22.4s, v8.4s, v0.s[3]  \n"
									"fmla   v23.4s, v9.4s, v0.s[3]  \n"

									"subs   w4, w4, #1              \n"

									"fmla   v24.4s, v8.4s, v1.s[0]  \n"
									"fmla   v25.4s, v9.4s, v1.s[0]  \n"
									"fmla   v26.4s, v8.4s, v1.s[1]  \n"
									"fmla   v27.4s, v9.4s, v1.s[1]  \n"
									"fmla   v28.4s, v8.4s, v1.s[2]  \n"
									"fmla   v29.4s, v9.4s, v1.s[2]  \n"
									"fmla   v30.4s, v8.4s, v1.s[3]  \n"
									"fmla   v31.4s, v9.4s, v1.s[3]  \n"

									"bne    2b                      \n"

									"3:                             \n"

									"st1    {v16.4s, v17.4s}, [%0], #32 \n"
									"st1    {v18.4s, v19.4s}, [%1], #32 \n"
									"st1    {v20.4s, v21.4s}, [%2], #32 \n"
									"st1    {v22.4s, v23.4s}, [%3], #32 \n"
									"st1    {v24.4s, v25.4s}, [%4], #32 \n"
									"st1    {v26.4s, v27.4s}, [%5], #32 \n"
									"st1    {v28.4s, v29.4s}, [%6], #32 \n"
									"st1    {v30.4s, v31.4s}, [%7], #32 \n"

									: "=r"(output0_tm), // %0
									"=r"(output1_tm), // %1
									"=r"(output2_tm), // %2
									"=r"(output3_tm), // %3
									"=r"(output4_tm), // %4
									"=r"(output5_tm), // %5
									"=r"(output6_tm), // %6
									"=r"(output7_tm), // %7
									"=r"(bb2p0),      // %8
									"=r"(ktm0)        // %9
									: "0"(output0_tm),
									"1"(output1_tm),
									"2"(output2_tm),
									"3"(output3_tm),
									"4"(output4_tm),
									"5"(output5_tm),
									"6"(output6_tm),
									"7"(output7_tm),
									"8"(bb2p0),
									"9"(ktm0),
									"r"(inch)         // %20
									: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
									);
							}
							for (; i + 3 < tiles; i += 4)
							{
								const float* bb2p0 = bb2 + (i / 8 + (i % 8) / 4) * bottom_tm2_w;

								const float* ktm0 = kernel_tm0 + (r)* kernel_tm_w;

								asm volatile(
									"eor    v16.16b, v16.16b, v16.16b  \n"
									"eor    v17.16b, v17.16b, v17.16b  \n"
									"eor    v18.16b, v18.16b, v18.16b  \n"
									"eor    v19.16b, v19.16b, v19.16b  \n"
									"eor    v20.16b, v20.16b, v20.16b  \n"
									"eor    v21.16b, v21.16b, v21.16b  \n"
									"eor    v22.16b, v22.16b, v22.16b  \n"
									"eor    v23.16b, v23.16b, v23.16b  \n"

									// inch loop
									"lsr    w4, %w20, #2            \n"// w4 = nn = inch >> 2
									"cmp    w4, #0                  \n"
									"beq    1f                      \n"

									"0:                             \n"

									"prfm   pldl1keep, [%8, #512]   \n"
									"ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%8], #64 \n"

									"prfm   pldl1keep, [%9, #512]   \n"
									"ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64   \n"

									"fmla   v16.4s, v8.4s, v0.s[0]  \n"
									"fmla   v17.4s, v8.4s, v0.s[1]  \n"
									"fmla   v18.4s, v8.4s, v0.s[2]  \n"
									"fmla   v19.4s, v8.4s, v0.s[3]  \n"
									"fmla   v20.4s, v8.4s, v1.s[0]  \n"
									"fmla   v21.4s, v8.4s, v1.s[1]  \n"
									"fmla   v22.4s, v8.4s, v1.s[2]  \n"
									"fmla   v23.4s, v8.4s, v1.s[3]  \n"

									"prfm   pldl1keep, [%9, #512]   \n"
									"ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%9], #64   \n"

									"fmla   v16.4s, v9.4s, v2.s[0]  \n"
									"fmla   v17.4s, v9.4s, v2.s[1]  \n"
									"fmla   v18.4s, v9.4s, v2.s[2]  \n"
									"fmla   v19.4s, v9.4s, v2.s[3]  \n"
									"fmla   v20.4s, v9.4s, v3.s[0]  \n"
									"fmla   v21.4s, v9.4s, v3.s[1]  \n"
									"fmla   v22.4s, v9.4s, v3.s[2]  \n"
									"fmla   v23.4s, v9.4s, v3.s[3]  \n"

									"fmla   v16.4s, v10.4s, v4.s[0] \n"
									"fmla   v17.4s, v10.4s, v4.s[1] \n"
									"fmla   v18.4s, v10.4s, v4.s[2] \n"
									"fmla   v19.4s, v10.4s, v4.s[3] \n"
									"fmla   v20.4s, v10.4s, v5.s[0] \n"
									"fmla   v21.4s, v10.4s, v5.s[1] \n"
									"fmla   v22.4s, v10.4s, v5.s[2] \n"
									"fmla   v23.4s, v10.4s, v5.s[3] \n"

									"subs   w4, w4, #1              \n"

									"fmla   v16.4s, v11.4s, v6.s[0] \n"
									"fmla   v17.4s, v11.4s, v6.s[1] \n"
									"fmla   v18.4s, v11.4s, v6.s[2] \n"
									"fmla   v19.4s, v11.4s, v6.s[3] \n"
									"fmla   v20.4s, v11.4s, v7.s[0] \n"
									"fmla   v21.4s, v11.4s, v7.s[1] \n"
									"fmla   v22.4s, v11.4s, v7.s[2] \n"
									"fmla   v23.4s, v11.4s, v7.s[3] \n"

									"bne    0b                      \n"

									"1:                             \n"

									// remain loop
									"and    w4, %w20, #3            \n"// w4 = remain = tiles & 3;
									"cmp    w4, #0                  \n"
									"beq    3f                      \n"

									"2:                             \n"

									"prfm   pldl1keep, [%8, #128]   \n"
									"ld1    {v8.4s}, [%8], #16      \n"

									"prfm   pldl1keep, [%9, #256]   \n"
									"ld1    {v0.4s, v1.4s}, [%9], #32   \n"

									"fmla   v16.4s, v8.4s, v0.s[0]  \n"
									"fmla   v17.4s, v8.4s, v0.s[1]  \n"
									"fmla   v18.4s, v8.4s, v0.s[2]  \n"
									"fmla   v19.4s, v8.4s, v0.s[3]  \n"

									"subs   w4, w4, #1              \n"

									"fmla   v20.4s, v8.4s, v1.s[0]  \n"
									"fmla   v21.4s, v8.4s, v1.s[1]  \n"
									"fmla   v22.4s, v8.4s, v1.s[2]  \n"
									"fmla   v23.4s, v8.4s, v1.s[3]  \n"

									"bne    2b                      \n"

									"3:                             \n"

									"st1    {v16.4s}, [%0], #16     \n"
									"st1    {v17.4s}, [%1], #16     \n"
									"st1    {v18.4s}, [%2], #16     \n"
									"st1    {v19.4s}, [%3], #16     \n"
									"st1    {v20.4s}, [%4], #16     \n"
									"st1    {v21.4s}, [%5], #16     \n"
									"st1    {v22.4s}, [%6], #16     \n"
									"st1    {v23.4s}, [%7], #16     \n"

									: "=r"(output0_tm), // %0
									"=r"(output1_tm), // %1
									"=r"(output2_tm), // %2
									"=r"(output3_tm), // %3
									"=r"(output4_tm), // %4
									"=r"(output5_tm), // %5
									"=r"(output6_tm), // %6
									"=r"(output7_tm), // %7
									"=r"(bb2p0),      // %8
									"=r"(ktm0)        // %9
									: "0"(output0_tm),
									"1"(output1_tm),
									"2"(output2_tm),
									"3"(output3_tm),
									"4"(output4_tm),
									"5"(output5_tm),
									"6"(output6_tm),
									"7"(output7_tm),
									"8"(bb2p0),
									"9"(ktm0),
									"r"(inch)         // %20
									: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
									);
							}
							for (; i < tiles; i++)
							{
								const float* bb2p0 = bb2 + (i / 8 + (i % 8) / 4 + i % 4) * bottom_tm2_w;

								const float* ktm0 = kernel_tm0 + (r)* kernel_tm_w;

								float32x4_t _sum0123 = vdupq_n_f32(0.f);
								float32x4_t _sum4567 = vdupq_n_f32(0.f);

								int q = 0;
								for (; q + 3 < inch; q += 4)
								{
									//                         asm volatile("prfm pldl1keep, [%0, #128] \n" : :"r"(bb2p0) :);
									float32x4_t _bb2p0 = vld1q_f32(bb2p0);
									bb2p0 += 4;

									//                         asm volatile("prfm pldl1keep, [%0, #512] \n" : :"r"(ktm0) :);
									float32x4_t _ktm0 = vld1q_f32(ktm0 + 0);
									float32x4_t _ktm1 = vld1q_f32(ktm0 + 4);
									float32x4_t _ktm2 = vld1q_f32(ktm0 + 8);
									float32x4_t _ktm3 = vld1q_f32(ktm0 + 12);
									ktm0 += 16;

									_sum0123 = vmlaq_laneq_f32(_sum0123, _ktm0, _bb2p0, 0);
									_sum4567 = vmlaq_laneq_f32(_sum4567, _ktm1, _bb2p0, 0);
									_sum0123 = vmlaq_laneq_f32(_sum0123, _ktm2, _bb2p0, 1);
									_sum4567 = vmlaq_laneq_f32(_sum4567, _ktm3, _bb2p0, 1);

									//                         asm volatile("prfm pldl1keep, [%0, #512] \n" : :"r"(ktm0) :);
									float32x4_t _ktm4 = vld1q_f32(ktm0 + 0);
									float32x4_t _ktm5 = vld1q_f32(ktm0 + 4);
									float32x4_t _ktm6 = vld1q_f32(ktm0 + 8);
									float32x4_t _ktm7 = vld1q_f32(ktm0 + 12);
									ktm0 += 16;

									_sum0123 = vmlaq_laneq_f32(_sum0123, _ktm4, _bb2p0, 2);
									_sum4567 = vmlaq_laneq_f32(_sum4567, _ktm5, _bb2p0, 2);
									_sum0123 = vmlaq_laneq_f32(_sum0123, _ktm6, _bb2p0, 3);
									_sum4567 = vmlaq_laneq_f32(_sum4567, _ktm7, _bb2p0, 3);
								}

								for (; q < inch; q++)
								{
									float32x4_t _bb2p0 = vld1q_dup_f32(bb2p0);
									float32x4_t _ktm0123 = vld1q_f32(ktm0 + 0);
									float32x4_t _ktm4567 = vld1q_f32(ktm0 + 4);

									_sum0123 = vmlaq_f32(_sum0123, _bb2p0, _ktm0123);
									_sum4567 = vmlaq_f32(_sum4567, _bb2p0, _ktm4567);

									bb2p0 += 1;
									ktm0 += 8;
								}

								float sum0 = vgetq_lane_f32(_sum0123, 0);
								float sum1 = vgetq_lane_f32(_sum0123, 1);
								float sum2 = vgetq_lane_f32(_sum0123, 2);
								float sum3 = vgetq_lane_f32(_sum0123, 3);
								float sum4 = vgetq_lane_f32(_sum4567, 0);
								float sum5 = vgetq_lane_f32(_sum4567, 1);
								float sum6 = vgetq_lane_f32(_sum4567, 2);
								float sum7 = vgetq_lane_f32(_sum4567, 3);

								output0_tm[0] = sum0;
								output1_tm[0] = sum1;
								output2_tm[0] = sum2;
								output3_tm[0] = sum3;
								output4_tm[0] = sum4;
								output5_tm[0] = sum5;
								output6_tm[0] = sum6;
								output7_tm[0] = sum7;

								output0_tm += 1;
								output1_tm += 1;
								output2_tm += 1;
								output3_tm += 1;
								output4_tm += 1;
								output5_tm += 1;
								output6_tm += 1;
								output7_tm += 1;
							}
						}
					}
#endif // __aarch64__

					nn_outch = (outch - remain_outch_start) >> 2;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
					for (int pp = 0; pp < nn_outch; pp++)
					{
						int p = remain_outch_start + pp * 4;

#if __ARM_NEON && __aarch64__
						const float *kernel_tm0 = kernel_tm_data + (p / 8 + (p % 8) / 4) * kernel_tm_cstep;
#else
						const float *kernel_tm0 = kernel_tm_data + (p / 4) * kernel_tm_cstep;
#endif

						float *out0_tm = top_tm_data + (p + 0) * top_tm_cstep;
						float *out1_tm = top_tm_data + (p + 1) * top_tm_cstep;
						float *out2_tm = top_tm_data + (p + 2) * top_tm_cstep;
						float *out3_tm = top_tm_data + (p + 3) * top_tm_cstep;

						float* output0_tm = out0_tm;
						float* output1_tm = out1_tm;
						float* output2_tm = out2_tm;
						float* output3_tm = out3_tm;

						for (int r = 0; r < 64; r++)
						{
							const float *bb2 = bottom_tm2_data + (r)* bottom_tm2_cstep;

							// tile
							int i = 0;
							for (; i + 7 < tiles; i += 8)
							{
								const float* bb2p0 = bb2 + (i / 8) * bottom_tm2_w;

								const float* ktm0 = kernel_tm0 + (r)* kernel_tm_w;
#if __ARM_NEON
#if __aarch64__
								asm volatile(
									"eor    v8.16b, v8.16b, v8.16b     \n"
									"eor    v9.16b, v9.16b, v9.16b     \n"
									"eor    v10.16b, v10.16b, v10.16b  \n"
									"eor    v11.16b, v11.16b, v11.16b  \n"
									"eor    v12.16b, v12.16b, v12.16b  \n"
									"eor    v13.16b, v13.16b, v13.16b  \n"
									"eor    v14.16b, v14.16b, v14.16b  \n"
									"eor    v15.16b, v15.16b, v15.16b  \n"

									// inch loop
									"lsr    w4, %w12, #2            \n"// w4 = nn = inch >> 2
									"cmp    w4, #0                  \n"
									"beq    1f                      \n"

									"0:                             \n"

									"prfm   pldl1keep, [%4, #512]   \n"
									"ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64     \n"

									"prfm   pldl1keep, [%5, #512]   \n"
									"ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64     \n"

									"fmla   v8.4s, v4.4s, v0.s[0]   \n"
									"fmla   v9.4s, v5.4s, v0.s[0]   \n"
									"fmla   v10.4s, v4.4s, v0.s[1]  \n"
									"fmla   v11.4s, v5.4s, v0.s[1]  \n"
									"fmla   v12.4s, v4.4s, v0.s[2]  \n"
									"fmla   v13.4s, v5.4s, v0.s[2]  \n"
									"fmla   v14.4s, v4.4s, v0.s[3]  \n"
									"fmla   v15.4s, v5.4s, v0.s[3]  \n"

									"prfm   pldl1keep, [%4, #512]   \n"
									"ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%4], #64 \n"

									"fmla   v8.4s, v6.4s, v1.s[0]   \n"
									"fmla   v9.4s, v7.4s, v1.s[0]   \n"
									"fmla   v10.4s, v6.4s, v1.s[1]  \n"
									"fmla   v11.4s, v7.4s, v1.s[1]  \n"
									"fmla   v12.4s, v6.4s, v1.s[2]  \n"
									"fmla   v13.4s, v7.4s, v1.s[2]  \n"
									"fmla   v14.4s, v6.4s, v1.s[3]  \n"
									"fmla   v15.4s, v7.4s, v1.s[3]  \n"

									"fmla   v8.4s, v16.4s, v2.s[0]  \n"
									"fmla   v9.4s, v17.4s, v2.s[0]  \n"
									"fmla   v10.4s, v16.4s, v2.s[1] \n"
									"fmla   v11.4s, v17.4s, v2.s[1] \n"
									"fmla   v12.4s, v16.4s, v2.s[2] \n"
									"fmla   v13.4s, v17.4s, v2.s[2] \n"
									"fmla   v14.4s, v16.4s, v2.s[3] \n"
									"fmla   v15.4s, v17.4s, v2.s[3] \n"

									"fmla   v8.4s, v18.4s, v3.s[0]  \n"
									"fmla   v9.4s, v19.4s, v3.s[0]  \n"
									"fmla   v10.4s, v18.4s, v3.s[1] \n"
									"fmla   v11.4s, v19.4s, v3.s[1] \n"
									"fmla   v12.4s, v18.4s, v3.s[2] \n"
									"fmla   v13.4s, v19.4s, v3.s[2] \n"
									"fmla   v14.4s, v18.4s, v3.s[3] \n"
									"fmla   v15.4s, v19.4s, v3.s[3] \n"

									"subs   w4, w4, #1              \n"
									"bne    0b                      \n"

									"1:                             \n"

									// remain loop
									"and    w4, %w12, #3            \n"// w4 = remain = tiles & 3;
									"cmp    w4, #0                  \n"
									"beq    3f                      \n"

									"2:                             \n"

									"prfm   pldl1keep, [%4, #256]   \n"
									"ld1    {v4.4s, v5.4s}, [%4], #32      \n"

									"prfm   pldl1keep, [%5, #128]   \n"
									"ld1    {v0.4s}, [%5], #16      \n"

									"fmla   v8.4s, v4.4s, v0.s[0]   \n"
									"fmla   v9.4s, v5.4s, v0.s[0]   \n"
									"fmla   v10.4s, v4.4s, v0.s[1]  \n"
									"fmla   v11.4s, v5.4s, v0.s[1]  \n"
									"fmla   v12.4s, v4.4s, v0.s[2]  \n"
									"fmla   v13.4s, v5.4s, v0.s[2]  \n"
									"fmla   v14.4s, v4.4s, v0.s[3]  \n"
									"fmla   v15.4s, v5.4s, v0.s[3]  \n"

									"subs   w4, w4, #1              \n"
									"bne    2b                      \n"

									"3:                             \n"

									"st1    {v8.4s, v9.4s}, [%0], #32       \n"
									"st1    {v10.4s, v11.4s}, [%1], #32     \n"
									"st1    {v12.4s, v13.4s}, [%2], #32     \n"
									"st1    {v14.4s, v15.4s}, [%3], #32     \n"

									: "=r"(output0_tm), // %0
									"=r"(output1_tm), // %1
									"=r"(output2_tm), // %2
									"=r"(output3_tm), // %3
									"=r"(bb2p0),      // %4
									"=r"(ktm0)        // %5
									: "0"(output0_tm),
									"1"(output1_tm),
									"2"(output2_tm),
									"3"(output3_tm),
									"4"(bb2p0),
									"5"(ktm0),
									"r"(inch)         // %12
									: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19"
									);
#else // __aarch64__
								asm volatile(
									"veor       q8, q8, q8      \n"
									"veor       q9, q9, q9      \n"
									"veor       q10, q10, q10   \n"
									"veor       q11, q11, q11   \n"
									"veor       q12, q12, q12   \n"
									"veor       q13, q13, q13   \n"
									"veor       q14, q14, q14   \n"
									"veor       q15, q15, q15   \n"

									// inch loop
									"lsr        r4, %12, #2     \n"// r4 = nn = inch >> 2
									"cmp        r4, #0          \n"
									"beq        1f              \n"

									"0:                         \n"

									"pld        [%4, #512]      \n"
									"vldm       %4!, {d8-d15}   \n"
									//                         "vld1.f32   {d8-d11}, [%4 :128]! \n"
									//                         "vld1.f32   {d12-d15}, [%4 :128]! \n"

									"pld        [%5, #512]      \n"
									"vldm       %5!, {d0-d7}    \n"
									//                         "vld1.f32   {d0-d3}, [%5 :128]!  \n"
									//                         "vld1.f32   {d4-d7}, [%5 :128]!  \n"

									"vmla.f32   q8, q4, d0[0]   \n"
									"vmla.f32   q9, q5, d0[0]   \n"
									"vmla.f32   q10, q4, d0[1]  \n"
									"vmla.f32   q11, q5, d0[1]  \n"
									"vmla.f32   q12, q4, d1[0]  \n"
									"vmla.f32   q13, q5, d1[0]  \n"
									"vmla.f32   q14, q4, d1[1]  \n"
									"vmla.f32   q15, q5, d1[1]  \n"

									"vmla.f32   q8, q6, d2[0]   \n"
									"vmla.f32   q9, q7, d2[0]   \n"
									"vmla.f32   q10, q6, d2[1]  \n"
									"vmla.f32   q11, q7, d2[1]  \n"
									"vmla.f32   q12, q6, d3[0]  \n"
									"vmla.f32   q13, q7, d3[0]  \n"
									"vmla.f32   q14, q6, d3[1]  \n"
									"vmla.f32   q15, q7, d3[1]  \n"

									"pld        [%4, #512]      \n"
									"vldm       %4!, {d8-d15}   \n"
									//                         "vld1.f32   {d8-d11}, [%4 :128]! \n"
									//                         "vld1.f32   {d12-d15}, [%4 :128]! \n"

									"vmla.f32   q8, q4, d4[0]   \n"
									"vmla.f32   q9, q5, d4[0]   \n"
									"vmla.f32   q10, q4, d4[1]  \n"
									"vmla.f32   q11, q5, d4[1]  \n"
									"vmla.f32   q12, q4, d5[0]  \n"
									"vmla.f32   q13, q5, d5[0]  \n"
									"vmla.f32   q14, q4, d5[1]  \n"
									"vmla.f32   q15, q5, d5[1]  \n"

									"subs       r4, r4, #1      \n"

									"vmla.f32   q8, q6, d6[0]   \n"
									"vmla.f32   q9, q7, d6[0]   \n"
									"vmla.f32   q10, q6, d6[1]  \n"
									"vmla.f32   q11, q7, d6[1]  \n"
									"vmla.f32   q12, q6, d7[0]  \n"
									"vmla.f32   q13, q7, d7[0]  \n"
									"vmla.f32   q14, q6, d7[1]  \n"
									"vmla.f32   q15, q7, d7[1]  \n"

									"bne        0b              \n"

									"1:                         \n"

									// remain loop
									"and        r4, %12, #3     \n"// r4 = remain = tiles & 3;
									"cmp        r4, #0          \n"
									"beq        3f              \n"

									"2:                         \n"

									"pld        [%4, #256]      \n"
									"vld1.f32   {d8-d11}, [%4 :128]! \n"

									"pld        [%5, #128]      \n"
									"vld1.f32   {d0-d1}, [%5 :128]!  \n"

									"vmla.f32   q8, q4, d0[0]   \n"
									"vmla.f32   q9, q5, d0[0]   \n"
									"vmla.f32   q10, q4, d0[1]  \n"
									"vmla.f32   q11, q5, d0[1]  \n"

									"subs       r4, r4, #1      \n"

									"vmla.f32   q12, q4, d1[0]  \n"
									"vmla.f32   q13, q5, d1[0]  \n"
									"vmla.f32   q14, q4, d1[1]  \n"
									"vmla.f32   q15, q5, d1[1]  \n"

									"bne        2b              \n"

									"3:                         \n"

									"vst1.f32   {d16-d19}, [%0]! \n"
									"vst1.f32   {d20-d23}, [%1]! \n"
									"vst1.f32   {d24-d27}, [%2]! \n"
									"vst1.f32   {d28-d31}, [%3]! \n"

									: "=r"(output0_tm), // %0
									"=r"(output1_tm), // %1
									"=r"(output2_tm), // %2
									"=r"(output3_tm), // %3
									"=r"(bb2p0),      // %4
									"=r"(ktm0)        // %5
									: "0"(output0_tm),
									"1"(output1_tm),
									"2"(output2_tm),
									"3"(output3_tm),
									"4"(bb2p0),
									"5"(ktm0),
									"r"(inch)         // %12
									: "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
									);
#endif // __aarch64__
#else
								float sum0_0 = 0.f;
								float sum0_1 = 0.f;
								float sum0_2 = 0.f;
								float sum0_3 = 0.f;
								float sum0_4 = 0.f;
								float sum0_5 = 0.f;
								float sum0_6 = 0.f;
								float sum0_7 = 0.f;

								float sum1_0 = 0.f;
								float sum1_1 = 0.f;
								float sum1_2 = 0.f;
								float sum1_3 = 0.f;
								float sum1_4 = 0.f;
								float sum1_5 = 0.f;
								float sum1_6 = 0.f;
								float sum1_7 = 0.f;

								float sum2_0 = 0.f;
								float sum2_1 = 0.f;
								float sum2_2 = 0.f;
								float sum2_3 = 0.f;
								float sum2_4 = 0.f;
								float sum2_5 = 0.f;
								float sum2_6 = 0.f;
								float sum2_7 = 0.f;

								float sum3_0 = 0.f;
								float sum3_1 = 0.f;
								float sum3_2 = 0.f;
								float sum3_3 = 0.f;
								float sum3_4 = 0.f;
								float sum3_5 = 0.f;
								float sum3_6 = 0.f;
								float sum3_7 = 0.f;

								for (int q = 0; q < inch; q++)
								{
									sum0_0 += bb2p0[0] * ktm0[0];
									sum0_1 += bb2p0[1] * ktm0[0];
									sum0_2 += bb2p0[2] * ktm0[0];
									sum0_3 += bb2p0[3] * ktm0[0];
									sum0_4 += bb2p0[4] * ktm0[0];
									sum0_5 += bb2p0[5] * ktm0[0];
									sum0_6 += bb2p0[6] * ktm0[0];
									sum0_7 += bb2p0[7] * ktm0[0];

									sum1_0 += bb2p0[0] * ktm0[1];
									sum1_1 += bb2p0[1] * ktm0[1];
									sum1_2 += bb2p0[2] * ktm0[1];
									sum1_3 += bb2p0[3] * ktm0[1];
									sum1_4 += bb2p0[4] * ktm0[1];
									sum1_5 += bb2p0[5] * ktm0[1];
									sum1_6 += bb2p0[6] * ktm0[1];
									sum1_7 += bb2p0[7] * ktm0[1];

									sum2_0 += bb2p0[0] * ktm0[2];
									sum2_1 += bb2p0[1] * ktm0[2];
									sum2_2 += bb2p0[2] * ktm0[2];
									sum2_3 += bb2p0[3] * ktm0[2];
									sum2_4 += bb2p0[4] * ktm0[2];
									sum2_5 += bb2p0[5] * ktm0[2];
									sum2_6 += bb2p0[6] * ktm0[2];
									sum2_7 += bb2p0[7] * ktm0[2];

									sum3_0 += bb2p0[0] * ktm0[3];
									sum3_1 += bb2p0[1] * ktm0[3];
									sum3_2 += bb2p0[2] * ktm0[3];
									sum3_3 += bb2p0[3] * ktm0[3];
									sum3_4 += bb2p0[4] * ktm0[3];
									sum3_5 += bb2p0[5] * ktm0[3];
									sum3_6 += bb2p0[6] * ktm0[3];
									sum3_7 += bb2p0[7] * ktm0[3];

									bb2p0 += 8;
									ktm0 += 4;
								}

								output0_tm[0] = sum0_0;
								output0_tm[1] = sum0_1;
								output0_tm[2] = sum0_2;
								output0_tm[3] = sum0_3;
								output0_tm[4] = sum0_4;
								output0_tm[5] = sum0_5;
								output0_tm[6] = sum0_6;
								output0_tm[7] = sum0_7;

								output1_tm[0] = sum1_0;
								output1_tm[1] = sum1_1;
								output1_tm[2] = sum1_2;
								output1_tm[3] = sum1_3;
								output1_tm[4] = sum1_4;
								output1_tm[5] = sum1_5;
								output1_tm[6] = sum1_6;
								output1_tm[7] = sum1_7;

								output2_tm[0] = sum2_0;
								output2_tm[1] = sum2_1;
								output2_tm[2] = sum2_2;
								output2_tm[3] = sum2_3;
								output2_tm[4] = sum2_4;
								output2_tm[5] = sum2_5;
								output2_tm[6] = sum2_6;
								output2_tm[7] = sum2_7;

								output3_tm[0] = sum3_0;
								output3_tm[1] = sum3_1;
								output3_tm[2] = sum3_2;
								output3_tm[3] = sum3_3;
								output3_tm[4] = sum3_4;
								output3_tm[5] = sum3_5;
								output3_tm[6] = sum3_6;
								output3_tm[7] = sum3_7;

								output0_tm += 8;
								output1_tm += 8;
								output2_tm += 8;
								output3_tm += 8;
#endif // __ARM_NEON
							}
							for (; i + 3 < tiles; i += 4)
							{
								const float* bb2p0 = bb2 + (i / 8 + (i % 8) / 4) * bottom_tm2_w;

								const float* ktm0 = kernel_tm0 + (r)* kernel_tm_w;
#if __ARM_NEON
#if __aarch64__
								asm volatile(
									"eor    v8.16b, v8.16b, v8.16b     \n"
									"eor    v9.16b, v9.16b, v9.16b     \n"
									"eor    v10.16b, v10.16b, v10.16b  \n"
									"eor    v11.16b, v11.16b, v11.16b  \n"

									// inch loop
									"lsr    w4, %w12, #2            \n"// w4 = nn = inch >> 2
									"cmp    w4, #0                  \n"
									"beq    1f                      \n"

									"0:                             \n"

									"prfm   pldl1keep, [%4, #512]   \n"
									"ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64     \n"

									"prfm   pldl1keep, [%5, #512]   \n"
									"ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64     \n"

									"fmla   v8.4s, v4.4s, v0.s[0]   \n"
									"fmla   v9.4s, v4.4s, v0.s[1]   \n"
									"fmla   v10.4s, v4.4s, v0.s[2]  \n"
									"fmla   v11.4s, v4.4s, v0.s[3]  \n"

									"fmla   v8.4s, v5.4s, v1.s[0]   \n"
									"fmla   v9.4s, v5.4s, v1.s[1]   \n"
									"fmla   v10.4s, v5.4s, v1.s[2]  \n"
									"fmla   v11.4s, v5.4s, v1.s[3]  \n"

									"fmla   v8.4s, v6.4s, v2.s[0]   \n"
									"fmla   v9.4s, v6.4s, v2.s[1]   \n"
									"fmla   v10.4s, v6.4s, v2.s[2]  \n"
									"fmla   v11.4s, v6.4s, v2.s[3]  \n"

									"fmla   v8.4s, v7.4s, v3.s[0]   \n"
									"fmla   v9.4s, v7.4s, v3.s[1]   \n"
									"fmla   v10.4s, v7.4s, v3.s[2]  \n"
									"fmla   v11.4s, v7.4s, v3.s[3]  \n"

									"subs   w4, w4, #1              \n"
									"bne    0b                      \n"

									"1:                             \n"

									// remain loop
									"and    w4, %w12, #3            \n"// w4 = remain = tiles & 3;
									"cmp    w4, #0                  \n"
									"beq    3f                      \n"

									"2:                             \n"

									"prfm   pldl1keep, [%4, #128]   \n"
									"ld1    {v4.4s}, [%4], #16      \n"

									"prfm   pldl1keep, [%5, #128]   \n"
									"ld1    {v0.4s}, [%5], #16      \n"

									"fmla   v8.4s, v4.4s, v0.s[0]   \n"
									"fmla   v9.4s, v4.4s, v0.s[1]   \n"
									"fmla   v10.4s, v4.4s, v0.s[2]  \n"
									"fmla   v11.4s, v4.4s, v0.s[3]  \n"

									"subs   w4, w4, #1              \n"
									"bne    2b                      \n"

									"3:                             \n"

									"st1    {v8.4s}, [%0], #16      \n"
									"st1    {v9.4s}, [%1], #16      \n"
									"st1    {v10.4s}, [%2], #16     \n"
									"st1    {v11.4s}, [%3], #16     \n"

									: "=r"(output0_tm), // %0
									"=r"(output1_tm), // %1
									"=r"(output2_tm), // %2
									"=r"(output3_tm), // %3
									"=r"(bb2p0),      // %4
									"=r"(ktm0)        // %5
									: "0"(output0_tm),
									"1"(output1_tm),
									"2"(output2_tm),
									"3"(output3_tm),
									"4"(bb2p0),
									"5"(ktm0),
									"r"(inch)         // %12
									: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"
									);
#else // __aarch64__
								asm volatile(
									"veor       q8, q8, q8      \n"
									"veor       q9, q9, q9      \n"
									"veor       q10, q10, q10   \n"
									"veor       q11, q11, q11   \n"

									// inch loop
									"lsr        r4, %12, #2     \n"// r4 = nn = inch >> 2
									"cmp        r4, #0          \n"
									"beq        1f              \n"

									"0:                         \n"

									"pld        [%4, #512]      \n"
									"vldm       %4!, {d8-d15}   \n"
									//                         "vld1.f32   {d8-d11}, [%4 :128]! \n"
									//                         "vld1.f32   {d12-d15}, [%4 :128]! \n"

									"pld        [%5, #512]      \n"
									"vldm       %5!, {d0-d7}    \n"
									//                         "vld1.f32   {d0-d3}, [%5 :128]!  \n"
									//                         "vld1.f32   {d4-d7}, [%5 :128]!  \n"

									"vmla.f32   q8, q4, d0[0]   \n"
									"vmla.f32   q9, q4, d0[1]   \n"
									"vmla.f32   q10, q4, d1[0]  \n"
									"vmla.f32   q11, q4, d1[1]  \n"

									"vmla.f32   q8, q5, d2[0]   \n"
									"vmla.f32   q9, q5, d2[1]   \n"
									"vmla.f32   q10, q5, d3[0]  \n"
									"vmla.f32   q11, q5, d3[1]  \n"

									"subs       r4, r4, #1      \n"

									"vmla.f32   q8, q6, d4[0]   \n"
									"vmla.f32   q9, q6, d4[1]   \n"
									"vmla.f32   q10, q6, d5[0]  \n"
									"vmla.f32   q11, q6, d5[1]  \n"

									"vmla.f32   q8, q7, d6[0]   \n"
									"vmla.f32   q9, q7, d6[1]   \n"
									"vmla.f32   q10, q7, d7[0]  \n"
									"vmla.f32   q11, q7, d7[1]  \n"

									"bne        0b              \n"

									"1:                         \n"

									// remain loop
									"and        r4, %12, #3     \n"// r4 = remain = tiles & 3;
									"cmp        r4, #0          \n"
									"beq        3f              \n"

									"2:                         \n"

									"pld        [%4, #128]      \n"
									"vld1.f32   {d8-d9}, [%4 :128]!  \n"

									"pld        [%5, #128]      \n"
									"vld1.f32   {d0-d1}, [%5 :128]!  \n"

									"subs       r4, r4, #1      \n"

									"vmla.f32   q8, q4, d0[0]   \n"
									"vmla.f32   q9, q4, d0[1]   \n"
									"vmla.f32   q10, q4, d1[0]  \n"
									"vmla.f32   q11, q4, d1[1]  \n"

									"bne        2b              \n"

									"3:                         \n"

									"vst1.f32   {d16-d17}, [%0]! \n"
									"vst1.f32   {d18-d19}, [%1]! \n"
									"vst1.f32   {d20-d21}, [%2]! \n"
									"vst1.f32   {d22-d23}, [%3]! \n"

									: "=r"(output0_tm), // %0
									"=r"(output1_tm), // %1
									"=r"(output2_tm), // %2
									"=r"(output3_tm), // %3
									"=r"(bb2p0),      // %4
									"=r"(ktm0)        // %5
									: "0"(output0_tm),
									"1"(output1_tm),
									"2"(output2_tm),
									"3"(output3_tm),
									"4"(bb2p0),
									"5"(ktm0),
									"r"(inch)         // %12
									: "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11"
									);
#endif // __aarch64__
#else
								float sum0_0 = 0.f;
								float sum0_1 = 0.f;
								float sum0_2 = 0.f;
								float sum0_3 = 0.f;

								float sum1_0 = 0.f;
								float sum1_1 = 0.f;
								float sum1_2 = 0.f;
								float sum1_3 = 0.f;

								float sum2_0 = 0.f;
								float sum2_1 = 0.f;
								float sum2_2 = 0.f;
								float sum2_3 = 0.f;

								float sum3_0 = 0.f;
								float sum3_1 = 0.f;
								float sum3_2 = 0.f;
								float sum3_3 = 0.f;

								for (int q = 0; q < inch; q++)
								{
									sum0_0 += bb2p0[0] * ktm0[0];
									sum0_1 += bb2p0[1] * ktm0[0];
									sum0_2 += bb2p0[2] * ktm0[0];
									sum0_3 += bb2p0[3] * ktm0[0];

									sum1_0 += bb2p0[0] * ktm0[1];
									sum1_1 += bb2p0[1] * ktm0[1];
									sum1_2 += bb2p0[2] * ktm0[1];
									sum1_3 += bb2p0[3] * ktm0[1];

									sum2_0 += bb2p0[0] * ktm0[2];
									sum2_1 += bb2p0[1] * ktm0[2];
									sum2_2 += bb2p0[2] * ktm0[2];
									sum2_3 += bb2p0[3] * ktm0[2];

									sum3_0 += bb2p0[0] * ktm0[3];
									sum3_1 += bb2p0[1] * ktm0[3];
									sum3_2 += bb2p0[2] * ktm0[3];
									sum3_3 += bb2p0[3] * ktm0[3];

									bb2p0 += 4;
									ktm0 += 4;
								}

								output0_tm[0] = sum0_0;
								output0_tm[1] = sum0_1;
								output0_tm[2] = sum0_2;
								output0_tm[3] = sum0_3;

								output1_tm[0] = sum1_0;
								output1_tm[1] = sum1_1;
								output1_tm[2] = sum1_2;
								output1_tm[3] = sum1_3;

								output2_tm[0] = sum2_0;
								output2_tm[1] = sum2_1;
								output2_tm[2] = sum2_2;
								output2_tm[3] = sum2_3;

								output3_tm[0] = sum3_0;
								output3_tm[1] = sum3_1;
								output3_tm[2] = sum3_2;
								output3_tm[3] = sum3_3;

								output0_tm += 4;
								output1_tm += 4;
								output2_tm += 4;
								output3_tm += 4;
#endif // __ARM_NEON
							}
							for (; i < tiles; i++)
							{
								const float* bb2p0 = bb2 + (i / 8 + (i % 8) / 4 + i % 4) * bottom_tm2_w;

								const float* ktm0 = kernel_tm0 + (r)* kernel_tm_w;

#if __ARM_NEON
								float32x4_t _sum0123 = vdupq_n_f32(0.f);

								int q = 0;
								for (; q + 3 < inch; q += 4)
								{
									//                         asm volatile("prfm pldl1keep, [%0, #128] \n" : :"r"(bb2p0) :);
									float32x4_t _bb2p0 = vld1q_f32(bb2p0);
									bb2p0 += 4;

									//                         asm volatile("prfm pldl1keep, [%0, #512] \n" : :"r"(ktm0) :);
									float32x4_t _ktm0 = vld1q_f32(ktm0 + 0);
									float32x4_t _ktm1 = vld1q_f32(ktm0 + 4);
									float32x4_t _ktm2 = vld1q_f32(ktm0 + 8);
									float32x4_t _ktm3 = vld1q_f32(ktm0 + 12);
									ktm0 += 16;

#if __aarch64__
									_sum0123 = vmlaq_laneq_f32(_sum0123, _ktm0, _bb2p0, 0);
									_sum0123 = vmlaq_laneq_f32(_sum0123, _ktm1, _bb2p0, 1);
									_sum0123 = vmlaq_laneq_f32(_sum0123, _ktm2, _bb2p0, 2);
									_sum0123 = vmlaq_laneq_f32(_sum0123, _ktm3, _bb2p0, 3);
#else
									_sum0123 = vmlaq_lane_f32(_sum0123, _ktm0, vget_low_f32(_bb2p0), 0);
									_sum0123 = vmlaq_lane_f32(_sum0123, _ktm1, vget_low_f32(_bb2p0), 1);
									_sum0123 = vmlaq_lane_f32(_sum0123, _ktm2, vget_high_f32(_bb2p0), 0);
									_sum0123 = vmlaq_lane_f32(_sum0123, _ktm3, vget_high_f32(_bb2p0), 1);
#endif // __aarch64__
								}

								for (; q < inch; q++)
								{
									float32x4_t _bb2p0 = vld1q_dup_f32(bb2p0);
									float32x4_t _ktm0 = vld1q_f32(ktm0);

									_sum0123 = vmlaq_f32(_sum0123, _bb2p0, _ktm0);

									bb2p0 += 1;
									ktm0 += 4;
								}

								float sum0 = vgetq_lane_f32(_sum0123, 0);
								float sum1 = vgetq_lane_f32(_sum0123, 1);
								float sum2 = vgetq_lane_f32(_sum0123, 2);
								float sum3 = vgetq_lane_f32(_sum0123, 3);
#else
								float sum0 = 0.f;
								float sum1 = 0.f;
								float sum2 = 0.f;
								float sum3 = 0.f;

								for (int q = 0; q < inch; q++)
								{
									sum0 += bb2p0[0] * ktm0[0];
									sum1 += bb2p0[0] * ktm0[1];
									sum2 += bb2p0[0] * ktm0[2];
									sum3 += bb2p0[0] * ktm0[3];

									bb2p0 += 1;
									ktm0 += 4;
								}

								output0_tm[0] = sum0;
								output1_tm[0] = sum1;
								output2_tm[0] = sum2;
								output3_tm[0] = sum3;
#endif // __ARM_NEON

								output0_tm += 1;
								output1_tm += 1;
								output2_tm += 1;
								output3_tm += 1;
							}
						}
					}

					remain_outch_start += nn_outch << 2;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
					for (int p = remain_outch_start; p < outch; p++)
					{
#if __ARM_NEON && __aarch64__
						const float *kernel_tm0 = kernel_tm_data + (p / 8 + (p % 8) / 4 + p % 4) * kernel_tm_cstep;
#else
						const float *kernel_tm0 = kernel_tm_data + (p / 4 + p % 4) * kernel_tm_cstep;
#endif

						float *out0_tm = top_tm_data + (p)* top_tm_cstep;

						float* output0_tm = out0_tm;

						for (int r = 0; r < 64; r++)
						{
							const float *bb2 = bottom_tm2_data + (r)* bottom_tm2_cstep;

							// tile
							int i = 0;
							for (; i + 7 < tiles; i += 8)
							{
								const float* bb2p0 = bb2 + (i / 8) * bottom_tm2_w;

								const float* ktm0 = kernel_tm0 + (r)* kernel_tm_w;
#if __ARM_NEON
#if __aarch64__
								asm volatile(
									"eor    v8.16b, v8.16b, v8.16b     \n"
									"eor    v9.16b, v9.16b, v9.16b     \n"

									// inch loop
									"lsr    w4, %w6, #2             \n"// w4 = nn = inch >> 2
									"cmp    w4, #0                  \n"
									"beq    1f                      \n"

									"0:                             \n"

									"prfm   pldl1keep, [%1, #512]   \n"
									"ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64     \n"

									"prfm   pldl1keep, [%2, #128]   \n"
									"ld1    {v0.4s}, [%2], #16      \n"

									"fmla   v8.4s, v4.4s, v0.s[0]   \n"
									"fmla   v9.4s, v5.4s, v0.s[0]   \n"
									"fmla   v8.4s, v6.4s, v0.s[1]   \n"
									"fmla   v9.4s, v7.4s, v0.s[1]   \n"

									"prfm   pldl1keep, [%1, #512]   \n"
									"ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%1], #64 \n"

									"fmla   v8.4s, v12.4s, v0.s[2]  \n"
									"fmla   v9.4s, v13.4s, v0.s[2]  \n"
									"fmla   v8.4s, v14.4s, v0.s[3]  \n"
									"fmla   v9.4s, v15.4s, v0.s[3]  \n"

									"subs   w4, w4, #1              \n"
									"bne    0b                      \n"

									"1:                             \n"

									// remain loop
									"and    w4, %w6, #3             \n"// w4 = remain = tiles & 3;
									"cmp    w4, #0                  \n"
									"beq    3f                      \n"

									"2:                             \n"

									"prfm   pldl1keep, [%1, #256]   \n"
									"ld1    {v4.4s, v5.4s}, [%1], #32      \n"

									"prfm   pldl1keep, [%2, #32]    \n"
									"ld1r   {v0.4s}, [%2], #4       \n"

									"fmla   v8.4s, v4.4s, v0.4s     \n"
									"fmla   v9.4s, v5.4s, v0.4s     \n"

									"subs   w4, w4, #1              \n"
									"bne    2b                      \n"

									"3:                             \n"

									"st1    {v8.4s, v9.4s}, [%0], #32       \n"

									: "=r"(output0_tm), // %0
									"=r"(bb2p0),      // %1
									"=r"(ktm0)        // %2
									: "0"(output0_tm),
									"1"(bb2p0),
									"2"(ktm0),
									"r"(inch)         // %6
									: "cc", "memory", "x4", "v0", "v4", "v5", "v6", "v7", "v8", "v9", "v12", "v13", "v14", "v15"
									);
#else // __aarch64__
								asm volatile(
									"veor       q8, q8, q8          \n"
									"veor       q9, q9, q9          \n"

									// inch loop
									"lsr        r4, %6, #2          \n"// r4 = nn = inch >> 2
									"cmp        r4, #0              \n"
									"beq        1f                  \n"

									"0:                             \n"

									"pld        [%1, #512]          \n"
									"vldm       %1!, {d8-d15}       \n"
									//                         "vld1.f32   {d8-d11}, [%1 :128]! \n"
									//                         "vld1.f32   {d12-d15}, [%1 :128]! \n"

									"pld        [%2, #128]          \n"
									"vld1.f32   {d0-d1}, [%2 :128]! \n"

									"vmla.f32   q8, q4, d0[0]       \n"
									"vmla.f32   q9, q5, d0[0]       \n"
									"vmla.f32   q8, q6, d0[1]       \n"
									"vmla.f32   q9, q7, d0[1]       \n"

									"pld        [%1, #512]          \n"
									"vldm       %1!, {d24-d31}      \n"
									//                         "vld1.f32   {d24-d27}, [%1 :128]! \n"
									//                         "vld1.f32   {d28-d31}, [%1 :128]! \n"

									"subs       r4, r4, #1          \n"

									"vmla.f32   q8, q12, d1[0]      \n"
									"vmla.f32   q9, q13, d1[0]      \n"
									"vmla.f32   q8, q14, d1[1]      \n"
									"vmla.f32   q9, q15, d1[1]      \n"

									"bne        0b                  \n"

									"1:                             \n"

									// remain loop
									"and        r4, %6, #3          \n"// r4 = remain = tiles & 3;
									"cmp        r4, #0              \n"
									"beq        3f                  \n"

									"2:                             \n"

									"pld        [%1, #256]          \n"
									"vld1.f32   {d8-d11}, [%1 :128]! \n"

									"pld        [%2, #32]           \n"
									"vld1.f32   {d0[],d1[]}, [%2]!  \n"

									"subs       r4, r4, #1          \n"

									"vmla.f32   q8, q4, q0          \n"
									"vmla.f32   q9, q5, q0          \n"

									"bne        2b                  \n"

									"3:                             \n"

									"vst1.f32   {d16-d19}, [%0]!    \n"

									: "=r"(output0_tm), // %0
									"=r"(bb2p0),      // %1
									"=r"(ktm0)        // %2
									: "0"(output0_tm),
									"1"(bb2p0),
									"2"(ktm0),
									"r"(inch)         // %6
									: "cc", "memory", "r4", "q0", "q4", "q5", "q6", "q7", "q8", "q9", "q12", "q13", "q14", "q15"
									);
#endif // __aarch64__
#else
								float sum0 = 0.f;
								float sum1 = 0.f;
								float sum2 = 0.f;
								float sum3 = 0.f;
								float sum4 = 0.f;
								float sum5 = 0.f;
								float sum6 = 0.f;
								float sum7 = 0.f;

								for (int q = 0; q < inch; q++)
								{
									sum0 += bb2p0[0] * ktm0[0];
									sum1 += bb2p0[1] * ktm0[0];
									sum2 += bb2p0[2] * ktm0[0];
									sum3 += bb2p0[3] * ktm0[0];
									sum4 += bb2p0[4] * ktm0[0];
									sum5 += bb2p0[5] * ktm0[0];
									sum6 += bb2p0[6] * ktm0[0];
									sum7 += bb2p0[7] * ktm0[0];

									bb2p0 += 8;
									ktm0 += 1;
								}

								output0_tm[0] = sum0;
								output0_tm[1] = sum1;
								output0_tm[2] = sum2;
								output0_tm[3] = sum3;
								output0_tm[4] = sum4;
								output0_tm[5] = sum5;
								output0_tm[6] = sum6;
								output0_tm[7] = sum7;

								output0_tm += 8;
#endif // __ARM_NEON
							}
							for (; i + 3 < tiles; i += 4)
							{
								const float* bb2p0 = bb2 + (i / 8 + (i % 8) / 4) * bottom_tm2_w;

								const float* ktm0 = kernel_tm0 + (r)* kernel_tm_w;
#if __ARM_NEON
#if __aarch64__
								asm volatile(
									"eor    v8.16b, v8.16b, v8.16b     \n"

									// inch loop
									"lsr    w4, %w6, #2             \n"// w4 = nn = inch >> 2
									"cmp    w4, #0                  \n"
									"beq    1f                      \n"

									"0:                             \n"

									"prfm   pldl1keep, [%4, #512]   \n"
									"ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64     \n"

									"prfm   pldl1keep, [%5, #128]   \n"
									"ld1    {v0.4s}, [%5], #16      \n"

									"fmla   v8.4s, v4.4s, v0.s[0]   \n"
									"fmla   v8.4s, v5.4s, v0.s[1]   \n"
									"fmla   v8.4s, v6.4s, v0.s[2]   \n"
									"fmla   v8.4s, v7.4s, v0.s[3]   \n"

									"subs   w4, w4, #1              \n"
									"bne    0b                      \n"

									"1:                             \n"

									// remain loop
									"and    w4, %w6, #3             \n"// w4 = remain = tiles & 3;
									"cmp    w4, #0                  \n"
									"beq    3f                      \n"

									"2:                             \n"

									"prfm   pldl1keep, [%4, #128]   \n"
									"ld1    {v4.4s}, [%4], #16      \n"

									"prfm   pldl1keep, [%5, #32]    \n"
									"ld1r   {v0.4s}, [%5], #4       \n"

									"fmla   v8.4s, v4.4s, v0.4s     \n"

									"subs   w4, w4, #1              \n"
									"bne    2b                      \n"

									"3:                             \n"

									"st1    {v8.4s}, [%0], #16      \n"

									: "=r"(output0_tm), // %0
									"=r"(bb2p0),      // %1
									"=r"(ktm0)        // %2
									: "0"(output0_tm),
									"1"(bb2p0),
									"2"(ktm0),
									"r"(inch)         // %6
									: "cc", "memory", "x4", "v0", "v4", "v5", "v6", "v7", "v8"
									);
#else // __aarch64__
								asm volatile(
									"veor       q8, q8, q8          \n"

									// inch loop
									"lsr        r4, %6, #2          \n"// r4 = nn = inch >> 2
									"cmp        r4, #0              \n"
									"beq        1f                  \n"

									"0:                             \n"

									"pld        [%4, #512]          \n"
									"vldm       %4!, {d8-d15}       \n"
									//                         "vld1.f32   {d8-d11}, [%4 :128]! \n"
									//                         "vld1.f32   {d12-d15}, [%4 :128]! \n"

									"pld        [%5, #128]          \n"
									"vld1.f32   {d0-d1}, [%5 :128]! \n"

									"subs       r4, r4, #1          \n"

									"vmla.f32   q8, q4, d0[0]       \n"
									"vmla.f32   q8, q5, d0[1]       \n"
									"vmla.f32   q8, q6, d1[0]       \n"
									"vmla.f32   q8, q7, d1[1]       \n"

									"bne        0b                  \n"

									"1:                             \n"

									// remain loop
									"and        r4, %6, #3          \n"// r4 = remain = tiles & 3;
									"cmp        r4, #0              \n"
									"beq        3f                  \n"

									"2:                             \n"

									"pld        [%4, #128]          \n"
									"vld1.f32   {d8-d9}, [%4]!      \n"

									"pld        [%5, #32]           \n"
									"vld1.f32   {d0[],d1[]}, [%5]!  \n"

									"subs       r4, r4, #1          \n"

									"vmla.f32   q8, q4, q0          \n"

									"bne        2b                  \n"

									"3:                             \n"

									"vst1.f32   {d16-d17}, [%0]!    \n"

									: "=r"(output0_tm), // %0
									"=r"(bb2p0),      // %1
									"=r"(ktm0)        // %2
									: "0"(output0_tm),
									"1"(bb2p0),
									"2"(ktm0),
									"r"(inch)         // %6
									: "cc", "memory", "r4", "q0", "q4", "q5", "q6", "q7", "q8"
									);
#endif // __aarch64__
#else
								float sum0 = 0.f;
								float sum1 = 0.f;
								float sum2 = 0.f;
								float sum3 = 0.f;

								for (int q = 0; q < inch; q++)
								{
									sum0 += bb2p0[0] * ktm0[0];
									sum1 += bb2p0[1] * ktm0[0];
									sum2 += bb2p0[2] * ktm0[0];
									sum3 += bb2p0[3] * ktm0[0];

									bb2p0 += 4;
									ktm0 += 1;
								}

								output0_tm[0] = sum0;
								output0_tm[1] = sum1;
								output0_tm[2] = sum2;
								output0_tm[3] = sum3;

								output0_tm += 4;
#endif // __ARM_NEON
							}
							for (; i < tiles; i++)
							{
								const float* bb2p0 = bb2 + (i / 8 + (i % 8) / 4 + i % 4) * bottom_tm2_w;

								const float* ktm0 = kernel_tm0 + (r)* kernel_tm_w;

								int q = 0;
#if __ARM_NEON
								float32x4_t _sum0 = vdupq_n_f32(0.f);
								for (; q + 3 < inch; q += 4)
								{
									//                         asm volatile("prfm pldl1keep, [%0, #128] \n" : :"r"(bb2p0) :);
									float32x4_t _bb2p0 = vld1q_f32(bb2p0);
									bb2p0 += 4;

									float32x4_t _ktm0 = vld1q_f32(ktm0);
									ktm0 += 4;

									_sum0 = vmlaq_f32(_sum0, _bb2p0, _ktm0);
								}

#if __aarch64__
								float sum0 = vaddvq_f32(_sum0);
#else
								float32x2_t _ss0 = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
								float sum0 = vget_lane_f32(vpadd_f32(_ss0, _ss0), 0);
#endif // __aarch64__
#else
								float sum0 = 0.f;
#endif
								for (; q < inch; q++)
								{
									sum0 += bb2p0[0] * ktm0[0];

									bb2p0 += 1;
									ktm0 += 1;
								}

								output0_tm[0] = sum0;

								output0_tm += 1;
							}
						}
					}
				}
				// END dot

				// BEGIN transform output
				{
					//         const float otm[6][8] = {
					//             {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
					//             {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
					//             {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
					//             {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
					//             {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
					//             {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
					//         };

					// 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
					// 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
					// 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
					// 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
					// 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
					// 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)

#if __ARM_NEON
					const float coeff[4] = { 4.f, 8.f, 16.f, 32.f };
					float32x4_t _coeff = vld1q_f32(coeff);
#endif // __ARM_NEON

					float *top_bordered_data = top_bordered->mutable_cpu_data() + num_i * outch * top_borderd_cstep;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
					for (int p = 0; p < outch; p++)
					{
						const float *out0_tm = top_tm_data + (p)* top_tm_cstep;
						float *out0 = top_bordered_data + (p)* top_borderd_cstep;

						const float bias0 = this->bias_term_ ? bias[p] : 0.f;
#if __ARM_NEON
						float32x2_t _bias0 = vdup_n_f32(bias0);
#endif // __ARM_NEON

						float tmp[6][8];

						// tile
						for (int i = 0; i < outh / 6; i++)
						{
							for (int j = 0; j < outw / 6; j++)
							{
#if __ARM_NEON
#if __aarch64__
								const float* output0_tm0 = out0_tm + (i * w_tm / 8 + j) * top_tm_w;
								const float* output0_tm1 = out0_tm + (i * w_tm / 8 + j + tiles * 8) * top_tm_w;
								const float* output0_tm2 = out0_tm + (i * w_tm / 8 + j + tiles * 16) * top_tm_w;
								const float* output0_tm3 = out0_tm + (i * w_tm / 8 + j + tiles * 24) * top_tm_w;

								for (int m = 0; m + 3 < 8; m += 4)
								{
									float32x4_t _output0_tm_00;
									float32x4_t _output0_tm_11;
									float32x4_t _output0_tm_22;
									float32x4_t _output0_tm_33;
									float32x4_t _output0_tm_44;
									float32x4_t _output0_tm_55;
									float32x4_t _output0_tm_66;
									float32x4_t _output0_tm_77;

									_output0_tm_00 = vsetq_lane_f32(output0_tm0[0], _output0_tm_00, 0);
									output0_tm0 += top_tm_w * tiles;
									_output0_tm_00 = vsetq_lane_f32(output0_tm1[0], _output0_tm_00, 1);
									output0_tm1 += top_tm_w * tiles;
									_output0_tm_00 = vsetq_lane_f32(output0_tm2[0], _output0_tm_00, 2);
									output0_tm2 += top_tm_w * tiles;
									_output0_tm_00 = vsetq_lane_f32(output0_tm3[0], _output0_tm_00, 3);
									output0_tm3 += top_tm_w * tiles;

									_output0_tm_11 = vsetq_lane_f32(output0_tm0[0], _output0_tm_11, 0);
									output0_tm0 += top_tm_w * tiles;
									_output0_tm_11 = vsetq_lane_f32(output0_tm1[0], _output0_tm_11, 1);
									output0_tm1 += top_tm_w * tiles;
									_output0_tm_11 = vsetq_lane_f32(output0_tm2[0], _output0_tm_11, 2);
									output0_tm2 += top_tm_w * tiles;
									_output0_tm_11 = vsetq_lane_f32(output0_tm3[0], _output0_tm_11, 3);
									output0_tm3 += top_tm_w * tiles;

									_output0_tm_22 = vsetq_lane_f32(output0_tm0[0], _output0_tm_22, 0);
									output0_tm0 += top_tm_w * tiles;
									_output0_tm_22 = vsetq_lane_f32(output0_tm1[0], _output0_tm_22, 1);
									output0_tm1 += top_tm_w * tiles;
									_output0_tm_22 = vsetq_lane_f32(output0_tm2[0], _output0_tm_22, 2);
									output0_tm2 += top_tm_w * tiles;
									_output0_tm_22 = vsetq_lane_f32(output0_tm3[0], _output0_tm_22, 3);
									output0_tm3 += top_tm_w * tiles;

									_output0_tm_33 = vsetq_lane_f32(output0_tm0[0], _output0_tm_33, 0);
									output0_tm0 += top_tm_w * tiles;
									_output0_tm_33 = vsetq_lane_f32(output0_tm1[0], _output0_tm_33, 1);
									output0_tm1 += top_tm_w * tiles;
									_output0_tm_33 = vsetq_lane_f32(output0_tm2[0], _output0_tm_33, 2);
									output0_tm2 += top_tm_w * tiles;
									_output0_tm_33 = vsetq_lane_f32(output0_tm3[0], _output0_tm_33, 3);
									output0_tm3 += top_tm_w * tiles;

									_output0_tm_44 = vsetq_lane_f32(output0_tm0[0], _output0_tm_44, 0);
									output0_tm0 += top_tm_w * tiles;
									_output0_tm_44 = vsetq_lane_f32(output0_tm1[0], _output0_tm_44, 1);
									output0_tm1 += top_tm_w * tiles;
									_output0_tm_44 = vsetq_lane_f32(output0_tm2[0], _output0_tm_44, 2);
									output0_tm2 += top_tm_w * tiles;
									_output0_tm_44 = vsetq_lane_f32(output0_tm3[0], _output0_tm_44, 3);
									output0_tm3 += top_tm_w * tiles;

									_output0_tm_55 = vsetq_lane_f32(output0_tm0[0], _output0_tm_55, 0);
									output0_tm0 += top_tm_w * tiles;
									_output0_tm_55 = vsetq_lane_f32(output0_tm1[0], _output0_tm_55, 1);
									output0_tm1 += top_tm_w * tiles;
									_output0_tm_55 = vsetq_lane_f32(output0_tm2[0], _output0_tm_55, 2);
									output0_tm2 += top_tm_w * tiles;
									_output0_tm_55 = vsetq_lane_f32(output0_tm3[0], _output0_tm_55, 3);
									output0_tm3 += top_tm_w * tiles;

									_output0_tm_66 = vsetq_lane_f32(output0_tm0[0], _output0_tm_66, 0);
									output0_tm0 += top_tm_w * tiles;
									_output0_tm_66 = vsetq_lane_f32(output0_tm1[0], _output0_tm_66, 1);
									output0_tm1 += top_tm_w * tiles;
									_output0_tm_66 = vsetq_lane_f32(output0_tm2[0], _output0_tm_66, 2);
									output0_tm2 += top_tm_w * tiles;
									_output0_tm_66 = vsetq_lane_f32(output0_tm3[0], _output0_tm_66, 3);
									output0_tm3 += top_tm_w * tiles;

									_output0_tm_77 = vsetq_lane_f32(output0_tm0[0], _output0_tm_77, 0);
									_output0_tm_77 = vsetq_lane_f32(output0_tm1[0], _output0_tm_77, 1);
									_output0_tm_77 = vsetq_lane_f32(output0_tm2[0], _output0_tm_77, 2);
									_output0_tm_77 = vsetq_lane_f32(output0_tm3[0], _output0_tm_77, 3);

									float32x4_t _tmp024a = vaddq_f32(_output0_tm_11, _output0_tm_22);
									float32x4_t _tmp135a = vsubq_f32(_output0_tm_11, _output0_tm_22);

									float32x4_t _tmp024b = vaddq_f32(_output0_tm_33, _output0_tm_44);
									float32x4_t _tmp135b = vsubq_f32(_output0_tm_33, _output0_tm_44);

									float32x4_t _tmp024c = vaddq_f32(_output0_tm_55, _output0_tm_66);
									float32x4_t _tmp135c = vsubq_f32(_output0_tm_55, _output0_tm_66);

									float32x4_t _tmp0 = vaddq_f32(_output0_tm_00, _tmp024a);
									_tmp0 = vmlaq_lane_f32(_tmp0, _tmp024c, vget_high_f32(_coeff), 1);
									_tmp0 = vaddq_f32(_tmp0, _tmp024b);

									float32x4_t _tmp2 = vmlaq_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeff), 0);
									_tmp2 = vmlaq_lane_f32(_tmp2, _tmp024c, vget_low_f32(_coeff), 1);

									float32x4_t _tmp4 = vmlaq_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeff), 0);
									_tmp4 = vaddq_f32(_tmp4, _tmp024c);
									_tmp4 = vaddq_f32(_tmp4, _tmp024c);

									vst1q_f32(&tmp[0][m], _tmp0);
									vst1q_f32(&tmp[2][m], _tmp2);
									vst1q_f32(&tmp[4][m], _tmp4);

									float32x4_t _tmp1 = vmlaq_lane_f32(_tmp135a, _tmp135c, vget_high_f32(_coeff), 0);
									_tmp1 = vaddq_f32(_tmp1, _tmp135b);
									_tmp1 = vaddq_f32(_tmp1, _tmp135b);

									float32x4_t _tmp3 = vmlaq_lane_f32(_tmp135a, _tmp135b, vget_low_f32(_coeff), 1);
									_tmp3 = vmlaq_lane_f32(_tmp3, _tmp135c, vget_low_f32(_coeff), 0);

									float32x4_t _tmp5 = vaddq_f32(_output0_tm_77, _tmp135a);
									_tmp5 = vmlaq_lane_f32(_tmp5, _tmp135b, vget_high_f32(_coeff), 1);
									_tmp5 = vaddq_f32(_tmp5, _tmp135c);

									vst1q_f32(&tmp[1][m], _tmp1);
									vst1q_f32(&tmp[3][m], _tmp3);
									vst1q_f32(&tmp[5][m], _tmp5);

									output0_tm0 += top_tm_w * tiles * 25;
									output0_tm1 += top_tm_w * tiles * 25;
									output0_tm2 += top_tm_w * tiles * 25;
									output0_tm3 += top_tm_w * tiles * 25;
								}

								const float* t0 = tmp[0];
								const float* t1 = tmp[1];

								float* output0 = out0 + (i * 6) * top_bordered_w + j * 6;
								float* output1 = output0 + outw;

								for (int m = 0; m + 1 < 6; m += 2)
								{
									float32x4_t _t0_0123 = vld1q_f32(t0);
									float32x4_t _t0_4567 = vld1q_f32(t0 + 4);
									float32x4_t _t1_0123 = vld1q_f32(t1);
									float32x4_t _t1_4567 = vld1q_f32(t1 + 4);

									float32x4x2_t _t01_00221133 = vtrnq_f32(_t0_0123, _t1_0123);
									float32x4x2_t _t01_44665577 = vtrnq_f32(_t0_4567, _t1_4567);

									float32x2_t _t_00 = vget_low_f32(_t01_00221133.val[0]);
									float32x2_t _t_11 = vget_low_f32(_t01_00221133.val[1]);
									float32x2_t _t_22 = vget_high_f32(_t01_00221133.val[0]);
									float32x2_t _t_33 = vget_high_f32(_t01_00221133.val[1]);
									float32x2_t _t_44 = vget_low_f32(_t01_44665577.val[0]);
									float32x2_t _t_55 = vget_low_f32(_t01_44665577.val[1]);
									float32x2_t _t_66 = vget_high_f32(_t01_44665577.val[0]);
									float32x2_t _t_77 = vget_high_f32(_t01_44665577.val[1]);

									float32x2_t _tmp024a = vadd_f32(_t_11, _t_22);
									float32x2_t _tmp135a = vsub_f32(_t_11, _t_22);

									float32x2_t _tmp024b = vadd_f32(_t_33, _t_44);
									float32x2_t _tmp135b = vsub_f32(_t_33, _t_44);

									float32x2_t _tmp024c = vadd_f32(_t_55, _t_66);
									float32x2_t _tmp135c = vsub_f32(_t_55, _t_66);

									float32x2_t _output_0 = vadd_f32(_t_00, _tmp024a);
									_output_0 = vmla_lane_f32(_output_0, _tmp024c, vget_high_f32(_coeff), 1);
									_output_0 = vadd_f32(_output_0, _tmp024b);
									_output_0 = vadd_f32(_output_0, _bias0);

									float32x2_t _output_2 = vmla_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeff), 0);
									_output_2 = vmla_lane_f32(_output_2, _tmp024c, vget_low_f32(_coeff), 1);
									_output_2 = vadd_f32(_output_2, _bias0);

									float32x2_t _output_4 = vmla_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeff), 0);
									_output_4 = vadd_f32(_output_4, _tmp024c);
									_output_4 = vadd_f32(_output_4, _tmp024c);
									_output_4 = vadd_f32(_output_4, _bias0);

									output0[0] = vget_lane_f32(_output_0, 0);
									output1[0] = vget_lane_f32(_output_0, 1);
									output0[2] = vget_lane_f32(_output_2, 0);
									output1[2] = vget_lane_f32(_output_2, 1);
									output0[4] = vget_lane_f32(_output_4, 0);
									output1[4] = vget_lane_f32(_output_4, 1);

									float32x2_t _output_1 = vmla_lane_f32(_tmp135a, _tmp135c, vget_high_f32(_coeff), 0);
									_output_1 = vadd_f32(_output_1, _tmp135b);
									_output_1 = vadd_f32(_output_1, _tmp135b);
									_output_1 = vadd_f32(_output_1, _bias0);

									float32x2_t _output_3 = vmla_lane_f32(_tmp135a, _tmp135b, vget_low_f32(_coeff), 1);
									_output_3 = vmla_lane_f32(_output_3, _tmp135c, vget_low_f32(_coeff), 0);
									_output_3 = vadd_f32(_output_3, _bias0);

									float32x2_t _output_5 = vadd_f32(_t_77, _tmp135a);
									_output_5 = vmla_lane_f32(_output_5, _tmp135b, vget_high_f32(_coeff), 1);
									_output_5 = vadd_f32(_output_5, _tmp135c);
									_output_5 = vadd_f32(_output_5, _bias0);

									output0[1] = vget_lane_f32(_output_1, 0);
									output1[1] = vget_lane_f32(_output_1, 1);
									output0[3] = vget_lane_f32(_output_3, 0);
									output1[3] = vget_lane_f32(_output_3, 1);
									output0[5] = vget_lane_f32(_output_5, 0);
									output1[5] = vget_lane_f32(_output_5, 1);

									t0 += 8 * 2;
									t1 += 8 * 2;
									output0 += outw * 2;
									output1 += outw * 2;
								}
#else // __aarch64__
								const float* output0_tm0_0 = out0_tm + (i * w_tm / 8 + j) * top_tm_w;
								const float* output0_tm1_0 = out0_tm + (i * w_tm / 8 + j + tiles * 8) * top_tm_w;
								const float* output0_tm2_0 = out0_tm + (i * w_tm / 8 + j + tiles * 16) * top_tm_w;
								const float* output0_tm3_0 = out0_tm + (i * w_tm / 8 + j + tiles * 24) * top_tm_w;
								const float* output0_tm0_4 = out0_tm + (i * w_tm / 8 + j + tiles * 32) * top_tm_w;
								const float* output0_tm1_4 = out0_tm + (i * w_tm / 8 + j + tiles * 40) * top_tm_w;
								const float* output0_tm2_4 = out0_tm + (i * w_tm / 8 + j + tiles * 48) * top_tm_w;
								const float* output0_tm3_4 = out0_tm + (i * w_tm / 8 + j + tiles * 56) * top_tm_w;

								float* t0 = tmp[0];
								float* t1 = tmp[1];

								//                     int step = top_tm_w * tiles * 2*4 *4;
								int step = top_tm_w * tiles * 4;

								asm volatile(

									// loop0
									//                         "vld1.f32   {d16-d17}, [%2], %21 \n"
									//                         "vld1.f32   {d18-d19}, [%3], %21 \n"
									//                         "vld1.f32   {d20-d21}, [%4], %21 \n"
									//                         "vld1.f32   {d22-d23}, [%5], %21 \n"
									//                         "vld1.f32   {d24-d25}, [%6], %21 \n"
									//                         "vld1.f32   {d26-d27}, [%7], %21 \n"
									//                         "vld1.f32   {d28-d29}, [%8], %21 \n"
									//                         "vld1.f32   {d30-d31}, [%9], %21 \n"

									//                         "vtrn.32    q8, q10             \n"
									//                         "vtrn.32    q9, q11             \n"
									//                         "vtrn.32    q12, q14            \n"
									//                         "vtrn.32    q13, q15            \n"

									//                         "vswp       d17, d24            \n"
									//                         "vswp       d19, d26            \n"
									//                         "vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
									//                         "vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77
									"vld1.f32   {d16[0]}, [%2], %21 \n"
									"vld1.f32   {d16[1]}, [%3], %21 \n"
									"vld1.f32   {d17[0]}, [%4], %21 \n"
									"vld1.f32   {d17[1]}, [%5], %21 \n"

									"vld1.f32   {d20[0]}, [%2], %21 \n"
									"vld1.f32   {d20[1]}, [%3], %21 \n"
									"vld1.f32   {d21[0]}, [%4], %21 \n"
									"vld1.f32   {d21[1]}, [%5], %21 \n"

									"vld1.f32   {d24[0]}, [%2], %21 \n"
									"vld1.f32   {d24[1]}, [%3], %21 \n"
									"vld1.f32   {d25[0]}, [%4], %21 \n"
									"vld1.f32   {d25[1]}, [%5], %21 \n"

									"vadd.f32   q2, q10, q12        \n"
									"vsub.f32   q3, q10, q12        \n"

									"vld1.f32   {d28[0]}, [%2], %21 \n"
									"vld1.f32   {d28[1]}, [%3], %21 \n"
									"vld1.f32   {d29[0]}, [%4], %21 \n"
									"vld1.f32   {d29[1]}, [%5], %21 \n"

									"vld1.f32   {d18[0]}, [%2], %21 \n"
									"vld1.f32   {d18[1]}, [%3], %21 \n"
									"vld1.f32   {d19[0]}, [%4], %21 \n"
									"vld1.f32   {d19[1]}, [%5], %21 \n"

									"vadd.f32   q4, q14, q9         \n"
									"vsub.f32   q5, q14, q9         \n"

									"vld1.f32   {d22[0]}, [%2], %21 \n"
									"vld1.f32   {d22[1]}, [%3], %21 \n"
									"vld1.f32   {d23[0]}, [%4], %21 \n"
									"vld1.f32   {d23[1]}, [%5], %21 \n"

									"vld1.f32   {d26[0]}, [%2], %21 \n"
									"vld1.f32   {d26[1]}, [%3], %21 \n"
									"vld1.f32   {d27[0]}, [%4], %21 \n"
									"vld1.f32   {d27[1]}, [%5], %21 \n"

									"vadd.f32   q6, q11, q13        \n"
									"vsub.f32   q7, q11, q13        \n"// spare q9 q10 q11 q12 q13 q14

									"vld1.f32   {d30[0]}, [%2]      \n"
									"vld1.f32   {d30[1]}, [%3]      \n"
									"vld1.f32   {d31[0]}, [%4]      \n"
									"vld1.f32   {d31[1]}, [%5]      \n"

									"vmov       q9, q3              \n"
									"vadd.f32   q8, q8, q2          \n"
									"vmla.f32   q9, q7, %f20[0]     \n"
									"vmov       q12, q2             \n"
									"vmov       q10, q2             \n"
									"vmov       q11, q3             \n"
									"vmla.f32   q12, q4, %f20[0]    \n"
									"vadd.f32   q15, q15, q3        \n"
									"vmla.f32   q8, q6, %f20[1]     \n"
									"vadd.f32   q9, q9, q5          \n"
									"vmla.f32   q10, q4, %e20[0]    \n"
									"vmla.f32   q11, q5, %e20[1]    \n"
									"vadd.f32   q12, q12, q6        \n"
									"vmla.f32   q15, q5, %f20[1]    \n"
									"vadd.f32   q8, q8, q4          \n"
									"vadd.f32   q9, q9, q5          \n"
									"vmla.f32   q10, q6, %e20[1]    \n"
									"vmla.f32   q11, q7, %e20[0]    \n"
									"vadd.f32   q12, q12, q6        \n"
									"vadd.f32   q15, q15, q7        \n"

									"vst1.f32   {d16-d17}, [%0]     \n"
									"add        %0, %0, #64         \n"

									"vst1.f32   {d18-d19}, [%1]     \n"
									"add        %1, %1, #64         \n"

									"vst1.f32   {d20-d21}, [%0]     \n"
									"add        %0, %0, #64         \n"

									"vst1.f32   {d22-d23}, [%1]     \n"
									"add        %1, %1, #64         \n"

									"vst1.f32   {d24-d25}, [%0]     \n"
									"sub        %0, %0, #112        \n"

									"vst1.f32   {d30-d31}, [%1]     \n"
									"sub        %1, %1, #112        \n"

									// loop1
									//                         "vld1.f32   {d16-d17}, [%2]     \n"
									//                         "vld1.f32   {d18-d19}, [%3]     \n"
									//                         "vld1.f32   {d20-d21}, [%4]     \n"
									//                         "vld1.f32   {d22-d23}, [%5]     \n"
									//                         "vld1.f32   {d24-d25}, [%6]     \n"
									//                         "vld1.f32   {d26-d27}, [%7]     \n"
									//                         "vld1.f32   {d28-d29}, [%8]     \n"
									//                         "vld1.f32   {d30-d31}, [%9]     \n"

									//                         "vtrn.32    q8, q10             \n"
									//                         "vtrn.32    q9, q11             \n"
									//                         "vtrn.32    q12, q14            \n"
									//                         "vtrn.32    q13, q15            \n"

									//                         "vswp       d17, d24            \n"
									//                         "vswp       d19, d26            \n"
									//                         "vswp       d21, d28            \n"//  q8 = 00   q9 = 44  q10 = 11  q11 = 55
									//                         "vswp       d23, d30            \n"// q12 = 22  q13 = 66  q14 = 33  q15 = 77
									"vld1.f32   {d16[0]}, [%6], %21 \n"
									"vld1.f32   {d16[1]}, [%7], %21 \n"
									"vld1.f32   {d17[0]}, [%8], %21 \n"
									"vld1.f32   {d17[1]}, [%9], %21 \n"

									"vld1.f32   {d20[0]}, [%6], %21 \n"
									"vld1.f32   {d20[1]}, [%7], %21 \n"
									"vld1.f32   {d21[0]}, [%8], %21 \n"
									"vld1.f32   {d21[1]}, [%9], %21 \n"

									"vld1.f32   {d24[0]}, [%6], %21 \n"
									"vld1.f32   {d24[1]}, [%7], %21 \n"
									"vld1.f32   {d25[0]}, [%8], %21 \n"
									"vld1.f32   {d25[1]}, [%9], %21 \n"

									"vadd.f32   q2, q10, q12        \n"
									"vsub.f32   q3, q10, q12        \n"

									"vld1.f32   {d28[0]}, [%6], %21 \n"
									"vld1.f32   {d28[1]}, [%7], %21 \n"
									"vld1.f32   {d29[0]}, [%8], %21 \n"
									"vld1.f32   {d29[1]}, [%9], %21 \n"

									"vld1.f32   {d18[0]}, [%6], %21 \n"
									"vld1.f32   {d18[1]}, [%7], %21 \n"
									"vld1.f32   {d19[0]}, [%8], %21 \n"
									"vld1.f32   {d19[1]}, [%9], %21 \n"

									"vadd.f32   q4, q14, q9         \n"
									"vsub.f32   q5, q14, q9         \n"

									"vld1.f32   {d22[0]}, [%6], %21 \n"
									"vld1.f32   {d22[1]}, [%7], %21 \n"
									"vld1.f32   {d23[0]}, [%8], %21 \n"
									"vld1.f32   {d23[1]}, [%9], %21 \n"

									"vld1.f32   {d26[0]}, [%6], %21 \n"
									"vld1.f32   {d26[1]}, [%7], %21 \n"
									"vld1.f32   {d27[0]}, [%8], %21 \n"
									"vld1.f32   {d27[1]}, [%9], %21 \n"

									"vadd.f32   q6, q11, q13        \n"
									"vsub.f32   q7, q11, q13        \n"// spare q9 q10 q11 q12 q13 q14

									"vld1.f32   {d30[0]}, [%6]      \n"
									"vld1.f32   {d30[1]}, [%7]      \n"
									"vld1.f32   {d31[0]}, [%8]      \n"
									"vld1.f32   {d31[1]}, [%9]      \n"

									"vmov       q9, q3              \n"
									"vadd.f32   q8, q8, q2          \n"
									"vmla.f32   q9, q7, %f20[0]     \n"
									"vmov       q12, q2             \n"
									"vmov       q10, q2             \n"
									"vmov       q11, q3             \n"
									"vmla.f32   q12, q4, %f20[0]    \n"
									"vadd.f32   q15, q15, q3        \n"
									"vmla.f32   q8, q6, %f20[1]     \n"
									"vadd.f32   q9, q9, q5          \n"
									"vmla.f32   q10, q4, %e20[0]    \n"
									"vmla.f32   q11, q5, %e20[1]    \n"
									"vadd.f32   q12, q12, q6        \n"
									"vmla.f32   q15, q5, %f20[1]    \n"
									"vadd.f32   q8, q8, q4          \n"
									"vadd.f32   q9, q9, q5          \n"
									"vmla.f32   q10, q6, %e20[1]    \n"
									"vmla.f32   q11, q7, %e20[0]    \n"
									"vadd.f32   q12, q12, q6        \n"
									"vadd.f32   q15, q15, q7        \n"

									"vst1.f32   {d16-d17}, [%0]     \n"
									"add        %0, %0, #64         \n"

									"vst1.f32   {d18-d19}, [%1]     \n"
									"add        %1, %1, #64         \n"

									"vst1.f32   {d20-d21}, [%0]     \n"
									"add        %0, %0, #64         \n"

									"vst1.f32   {d22-d23}, [%1]     \n"
									"add        %1, %1, #64         \n"

									"vst1.f32   {d24-d25}, [%0]     \n"

									"vst1.f32   {d30-d31}, [%1]     \n"

									: "=r"(t0),             // %0
									"=r"(t1),             // %1
									"=r"(output0_tm0_0),  // %2
									"=r"(output0_tm1_0),  // %3
									"=r"(output0_tm2_0),  // %4
									"=r"(output0_tm3_0),  // %5
									"=r"(output0_tm0_4),  // %6
									"=r"(output0_tm1_4),  // %7
									"=r"(output0_tm2_4),  // %8
									"=r"(output0_tm3_4)   // %9
									: "0"(t0),
									"1"(t1),
									"2"(output0_tm0_0),
									"3"(output0_tm1_0),
									"4"(output0_tm2_0),
									"5"(output0_tm3_0),
									"6"(output0_tm0_4),
									"7"(output0_tm1_4),
									"8"(output0_tm2_4),
									"9"(output0_tm3_4),
									"w"(_coeff),          // %20
									"r"(step)             // %21
									: "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
									);

								t0 = tmp[0];
								t1 = tmp[1];

								float* output0 = out0 + (i * 6) * top_bordered_w + j * 6;
								float* output1 = output0 + outw;

								int stepw = outw * 2 * 4;

								asm volatile(

									// loop0
									"vld1.f32   {d16-d19}, [%2]     \n"
									"vld1.f32   {d20-d23}, [%3]     \n"

									"add        %2, %2, #64         \n"
									"add        %3, %3, #64         \n"

									"vtrn.32    q8, q10             \n"// q8 = 0 2  q10 = 1 3
									"vtrn.32    q9, q11             \n"// q9 = 4 6  q11 = 5 7

									"vadd.f32   d4, d20, d17        \n"
									"vsub.f32   d5, d20, d17        \n"

									"vadd.f32   d6, d21, d18        \n"
									"vsub.f32   d7, d21, d18        \n"

									"vadd.f32   d8, d22, d19        \n"
									"vsub.f32   d9, d22, d19        \n"// spare d17 ~ d22

									"vmov       d20, d5             \n"
									"vmov       d18, d4             \n"

									"vadd.f32   d16, d16, d4        \n"
									"vmla.f32   d20, d9, %f8[0]     \n"
									"vmov       d17, d4             \n"
									"vmov       d21, d5             \n"
									"vmla.f32   d18, d6, %f8[0]     \n"
									"vadd.f32   d22, d23, d5        \n"

									"vmla.f32   d16, d8, %f8[1]     \n"
									"vadd.f32   d20, d20, d7        \n"
									"vmla.f32   d17, d6, %e8[0]     \n"
									"vmla.f32   d21, d7, %e8[1]     \n"
									"vadd.f32   d18, d18, d8        \n"
									"vmla.f32   d22, d7, %f8[1]     \n"

									"vadd.f32   d16, d16, d6        \n"
									"vadd.f32   d20, d20, d7        \n"
									"vmla.f32   d17, d8, %e8[1]     \n"
									"vmla.f32   d21, d9, %e8[0]     \n"
									"vadd.f32   d18, d18, d8        \n"
									"vadd.f32   d22, d22, d9        \n"

									"vadd.f32   d16, d16, %P9       \n"// _bias0
									"vadd.f32   d20, d20, %P9       \n"// _bias0
									"vadd.f32   d17, d17, %P9       \n"// _bias0
									"vadd.f32   d21, d21, %P9       \n"// _bias0
									"vadd.f32   d18, d18, %P9       \n"// _bias0
									"vadd.f32   d22, d22, %P9       \n"// _bias0

									"vtrn.f32   q8, q10             \n"
									"vtrn.f32   d18, d22            \n"

									"vst1.f32   {d16-d18}, [%0], %10 \n"
									"vst1.f32   {d20-d22}, [%1], %10 \n"

									// loop1
									"vld1.f32   {d16-d19}, [%2]     \n"
									"vld1.f32   {d20-d23}, [%3]     \n"

									"add        %2, %2, #64         \n"
									"add        %3, %3, #64         \n"

									"vtrn.32    q8, q10             \n"// q8 = 0 2  q10 = 1 3
									"vtrn.32    q9, q11             \n"// q9 = 4 6  q11 = 5 7

									"vadd.f32   d4, d20, d17        \n"
									"vsub.f32   d5, d20, d17        \n"

									"vadd.f32   d6, d21, d18        \n"
									"vsub.f32   d7, d21, d18        \n"

									"vadd.f32   d8, d22, d19        \n"
									"vsub.f32   d9, d22, d19        \n"// spare d17 ~ d22

									"vmov       d20, d5             \n"
									"vmov       d18, d4             \n"

									"vadd.f32   d16, d16, d4        \n"
									"vmla.f32   d20, d9, %f8[0]     \n"
									"vmov       d17, d4             \n"
									"vmov       d21, d5             \n"
									"vmla.f32   d18, d6, %f8[0]     \n"
									"vadd.f32   d22, d23, d5        \n"

									"vmla.f32   d16, d8, %f8[1]     \n"
									"vadd.f32   d20, d20, d7        \n"
									"vmla.f32   d17, d6, %e8[0]     \n"
									"vmla.f32   d21, d7, %e8[1]     \n"
									"vadd.f32   d18, d18, d8        \n"
									"vmla.f32   d22, d7, %f8[1]     \n"

									"vadd.f32   d16, d16, d6        \n"
									"vadd.f32   d20, d20, d7        \n"
									"vmla.f32   d17, d8, %e8[1]     \n"
									"vmla.f32   d21, d9, %e8[0]     \n"
									"vadd.f32   d18, d18, d8        \n"
									"vadd.f32   d22, d22, d9        \n"

									"vadd.f32   d16, d16, %P9       \n"// _bias0
									"vadd.f32   d20, d20, %P9       \n"// _bias0
									"vadd.f32   d17, d17, %P9       \n"// _bias0
									"vadd.f32   d21, d21, %P9       \n"// _bias0
									"vadd.f32   d18, d18, %P9       \n"// _bias0
									"vadd.f32   d22, d22, %P9       \n"// _bias0

									"vtrn.f32   q8, q10             \n"
									"vtrn.f32   d18, d22            \n"

									"vst1.f32   {d16-d18}, [%0], %10 \n"
									"vst1.f32   {d20-d22}, [%1], %10 \n"

									// loop2
									"vld1.f32   {d16-d19}, [%2]     \n"
									"vld1.f32   {d20-d23}, [%3]     \n"

									"add        %2, %2, #64         \n"
									"add        %3, %3, #64         \n"

									"vtrn.32    q8, q10             \n"// q8 = 0 2  q10 = 1 3
									"vtrn.32    q9, q11             \n"// q9 = 4 6  q11 = 5 7

									"vadd.f32   d4, d20, d17        \n"
									"vsub.f32   d5, d20, d17        \n"

									"vadd.f32   d6, d21, d18        \n"
									"vsub.f32   d7, d21, d18        \n"

									"vadd.f32   d8, d22, d19        \n"
									"vsub.f32   d9, d22, d19        \n"// spare d17 ~ d22

									"vmov       d20, d5             \n"
									"vmov       d18, d4             \n"

									"vadd.f32   d16, d16, d4        \n"
									"vmla.f32   d20, d9, %f8[0]     \n"
									"vmov       d17, d4             \n"
									"vmov       d21, d5             \n"
									"vmla.f32   d18, d6, %f8[0]     \n"
									"vadd.f32   d22, d23, d5        \n"

									"vmla.f32   d16, d8, %f8[1]     \n"
									"vadd.f32   d20, d20, d7        \n"
									"vmla.f32   d17, d6, %e8[0]     \n"
									"vmla.f32   d21, d7, %e8[1]     \n"
									"vadd.f32   d18, d18, d8        \n"
									"vmla.f32   d22, d7, %f8[1]     \n"

									"vadd.f32   d16, d16, d6        \n"
									"vadd.f32   d20, d20, d7        \n"
									"vmla.f32   d17, d8, %e8[1]     \n"
									"vmla.f32   d21, d9, %e8[0]     \n"
									"vadd.f32   d18, d18, d8        \n"
									"vadd.f32   d22, d22, d9        \n"

									"vadd.f32   d16, d16, %P9       \n"// _bias0
									"vadd.f32   d20, d20, %P9       \n"// _bias0
									"vadd.f32   d17, d17, %P9       \n"// _bias0
									"vadd.f32   d21, d21, %P9       \n"// _bias0
									"vadd.f32   d18, d18, %P9       \n"// _bias0
									"vadd.f32   d22, d22, %P9       \n"// _bias0

									"vtrn.f32   q8, q10             \n"
									"vtrn.f32   d18, d22            \n"

									"vst1.f32   {d16-d18}, [%0], %10 \n"
									"vst1.f32   {d20-d22}, [%1], %10 \n"

									: "=r"(output0),    // %0
									"=r"(output1),    // %1
									"=r"(t0),         // %2
									"=r"(t1)          // %3
									: "0"(output0),
									"1"(output1),
									"2"(t0),
									"3"(t1),
									"w"(_coeff),      // %8
									"w"(_bias0),      // %9
									"r"(stepw)        // %10
									: "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
									);
#endif // __aarch64__
#else
								const float* output0_tm_0 = out0_tm + (i * w_tm / 8 + j) * top_tm_w;
								const float* output0_tm_1 = out0_tm + (i * w_tm / 8 + j + tiles) * top_tm_w;
								const float* output0_tm_2 = out0_tm + (i * w_tm / 8 + j + tiles * 2) * top_tm_w;
								const float* output0_tm_3 = out0_tm + (i * w_tm / 8 + j + tiles * 3) * top_tm_w;
								const float* output0_tm_4 = out0_tm + (i * w_tm / 8 + j + tiles * 4) * top_tm_w;
								const float* output0_tm_5 = out0_tm + (i * w_tm / 8 + j + tiles * 5) * top_tm_w;
								const float* output0_tm_6 = out0_tm + (i * w_tm / 8 + j + tiles * 6) * top_tm_w;
								const float* output0_tm_7 = out0_tm + (i * w_tm / 8 + j + tiles * 7) * top_tm_w;

								for (int m = 0; m < 8; m++)
								{
									float tmp024a = output0_tm_1[0] + output0_tm_2[0];
									float tmp135a = output0_tm_1[0] - output0_tm_2[0];

									float tmp024b = output0_tm_3[0] + output0_tm_4[0];
									float tmp135b = output0_tm_3[0] - output0_tm_4[0];

									float tmp024c = output0_tm_5[0] + output0_tm_6[0];
									float tmp135c = output0_tm_5[0] - output0_tm_6[0];

									tmp[0][m] = output0_tm_0[0] + tmp024a + tmp024b + tmp024c * 32;
									tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 8;
									tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c;

									tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * 16;
									tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 4;
									tmp[5][m] = output0_tm_7[0] + tmp135a + tmp135b * 32 + tmp135c;

									output0_tm_0 += top_tm_w * tiles * 8;
									output0_tm_1 += top_tm_w * tiles * 8;
									output0_tm_2 += top_tm_w * tiles * 8;
									output0_tm_3 += top_tm_w * tiles * 8;
									output0_tm_4 += top_tm_w * tiles * 8;
									output0_tm_5 += top_tm_w * tiles * 8;
									output0_tm_6 += top_tm_w * tiles * 8;
									output0_tm_7 += top_tm_w * tiles * 8;
								}

								float* output0 = out0 + (i * 6) * top_bordered_w + j * 6;

								for (int m = 0; m < 6; m++)
								{
									const float* tmp0 = tmp[m];

									float tmp024a = tmp0[1] + tmp0[2];
									float tmp135a = tmp0[1] - tmp0[2];

									float tmp024b = tmp0[3] + tmp0[4];
									float tmp135b = tmp0[3] - tmp0[4];

									float tmp024c = tmp0[5] + tmp0[6];
									float tmp135c = tmp0[5] - tmp0[6];

									output0[0] = bias0 + tmp0[0] + tmp024a + tmp024b + tmp024c * 32;
									output0[2] = bias0 + tmp024a + tmp024b * 4 + tmp024c * 8;
									output0[4] = bias0 + tmp024a + tmp024b * 16 + tmp024c + tmp024c;

									output0[1] = bias0 + tmp135a + tmp135b + tmp135b + tmp135c * 16;
									output0[3] = bias0 + tmp135a + tmp135b * 8 + tmp135c * 4;
									output0[5] = bias0 + tmp0[7] + tmp135a + tmp135b * 32 + tmp135c;

									output0 += outw;
								}
#endif // __ARM_NEON
							}
						}
					}
				}
				// END transform output
			}

			// cut result pad
			cut_border_cpu(top_bordered, top, 0, top_bordered->height() - top->height(), 0, top_bordered->width() - top->width());
		}

		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv3x3s1_neon(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top)
		{
			int num = bottom->num();
			int w = bottom->width();
			int h = bottom->height();
			int inch = bottom->channels();
			int bottom_cstep = w * h;

			int outw = top->width();
			int outh = top->height();
			int outch = top->channels();
			int top_cstep = outw * outh;

			const float* kernel = this->weights_f32_[0]->cpu_data();

			const float* bias = nullptr;
			if (this->bias_term_)
				bias = this->weights_f32_[1]->cpu_data();

			for (int num_i = 0; num_i < num; num_i++)
			{
				const float *bottom_data = bottom->cpu_data() + num_i * inch * bottom_cstep;
				float *top_data = top->mutable_cpu_data() + num_i * outch * top_cstep;

				int nn_outch = outch >> 1;
				int remain_outch_start = nn_outch << 1;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
				for (int pp = 0; pp < nn_outch; pp++)
				{
					int p = pp * 2;

					float *out0 = top_data + (p)* top_cstep;
					float *out1 = top_data + (p + 1) * top_cstep;

					const float bias0 = this->bias_term_ ? bias[p] : 0.f;
					const float bias1 = this->bias_term_ ? bias[p + 1] : 0.f;

					fill(out0, top_cstep, bias0);
					fill(out1, top_cstep, bias1);

					const float* k0 = kernel + p * inch * 9;
					const float* k1 = kernel + (p + 1)*inch * 9;

					for (int q = 0; q < inch; q++)
					{
						float* outptr0 = out0;
						float* outptr1 = out1;
						float* outptr0n = outptr0 + outw;
						float* outptr1n = outptr1 + outw;

						const float* img0 = bottom_data + (q)* bottom_cstep;

						const float* r0 = img0;
						const float* r1 = img0 + w;
						const float* r2 = img0 + w * 2;
						const float* r3 = img0 + w * 3;

#if __ARM_NEON
						float32x4_t _k00 = vld1q_f32(k0);
						float32x4_t _k03 = vld1q_f32(k0 + 3);
						float32x4_t _k06 = vld1q_f32(k0 + 6);

						float32x4_t _k10 = vld1q_f32(k1);
						float32x4_t _k13 = vld1q_f32(k1 + 3);
						float32x4_t _k16 = vld1q_f32(k1 + 6);
#endif // __ARM_NEON

						int i = 0;

						for (; i + 1 < outh; i += 2)
						{
#if __ARM_NEON
							int nn = outw >> 2;
							int remain = outw & 3;
#else
							int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
							if (nn > 0)
							{
								asm volatile(
									"prfm   pldl1keep, [%5, #256]       \n"
									"ld1    {v8.4s, v9.4s}, [%5]        \n"// r0
									"add    %5, %5, #16                 \n"

									"prfm   pldl1keep, [%8, #256]       \n"
									"ld1    {v14.4s, v15.4s}, [%8]      \n"// r3
									"add    %8, %8, #16                 \n"

									"ext    v10.16b, v8.16b, v9.16b, #4 \n"
									"ext    v11.16b, v14.16b, v15.16b, #8 \n"

									"0:                                 \n"

									"prfm   pldl1keep, [%1, #128]       \n"
									"ld1    {v6.4s}, [%1]               \n"// _sum0

									"prfm   pldl1keep, [%2, #128]       \n"
									"ld1    {v7.4s}, [%2]               \n"// _sum1

									"fmla   v6.4s, v8.4s, %18.s[0]      \n"
									"fmla   v7.4s, v8.4s, %21.s[0]      \n"

									"prfm   pldl1keep, [%3, #128]       \n"
									"ld1    {v12.4s}, [%3]              \n"// _sum0n

									"prfm   pldl1keep, [%4, #128]       \n"
									"ld1    {v13.4s}, [%4]              \n"// _sum1n

									"fmla   v12.4s, v14.4s, %20.s[0]    \n"
									"fmla   v13.4s, v14.4s, %23.s[0]    \n"

									"ext    v8.16b, v8.16b, v9.16b, #8  \n"
									"ext    v9.16b, v14.16b, v15.16b, #4 \n"

									"fmla   v6.4s, v10.4s, %18.s[1]     \n"
									"fmla   v7.4s, v10.4s, %21.s[1]     \n"
									"fmla   v12.4s, v11.4s, %20.s[2]    \n"
									"fmla   v13.4s, v11.4s, %23.s[2]    \n"

									"prfm   pldl1keep, [%6, #256]       \n"
									"ld1    {v14.4s, v15.4s}, [%6]      \n"// r1
									"add    %6, %6, #16                 \n"

									"fmla   v6.4s, v8.4s, %18.s[2]      \n"
									"fmla   v7.4s, v8.4s, %21.s[2]      \n"
									"fmla   v12.4s, v9.4s, %20.s[1]     \n"
									"fmla   v13.4s, v9.4s, %23.s[1]     \n"

									"ext    v10.16b, v14.16b, v15.16b, #4 \n"

									"fmla   v6.4s, v14.4s, %19.s[0]     \n"
									"fmla   v7.4s, v14.4s, %22.s[0]     \n"
									"fmla   v12.4s, v14.4s, %18.s[0]    \n"
									"fmla   v13.4s, v14.4s, %21.s[0]    \n"

									"ext    v11.16b, v14.16b, v15.16b, #8 \n"

									"fmla   v6.4s, v10.4s, %19.s[1]     \n"
									"fmla   v7.4s, v10.4s, %22.s[1]     \n"
									"fmla   v12.4s, v10.4s, %18.s[1]    \n"
									"fmla   v13.4s, v10.4s, %21.s[1]    \n"

									"prfm   pldl1keep, [%7, #256]       \n"
									"ld1    {v8.4s, v9.4s}, [%7]        \n"// r2
									"add    %7, %7, #16                 \n"

									"fmla   v6.4s, v11.4s, %19.s[2]     \n"
									"fmla   v7.4s, v11.4s, %22.s[2]     \n"
									"fmla   v12.4s, v11.4s, %18.s[2]    \n"
									"fmla   v13.4s, v11.4s, %21.s[2]    \n"

									"ext    v10.16b, v8.16b, v9.16b, #4 \n"

									"fmla   v6.4s, v8.4s, %20.s[0]      \n"
									"fmla   v7.4s, v8.4s, %23.s[0]      \n"
									"fmla   v12.4s, v8.4s, %19.s[0]     \n"
									"fmla   v13.4s, v8.4s, %22.s[0]     \n"

									"ext    v11.16b, v8.16b, v9.16b, #8 \n"

									"fmla   v6.4s, v10.4s, %20.s[1]     \n"
									"fmla   v7.4s, v10.4s, %23.s[1]     \n"
									"fmla   v12.4s, v10.4s, %19.s[1]    \n"
									"fmla   v13.4s, v10.4s, %22.s[1]    \n"

									"prfm   pldl1keep, [%5, #256]       \n"
									"ld1    {v8.4s, v9.4s}, [%5]        \n"// r0
									"add    %5, %5, #16                 \n"

									"fmla   v6.4s, v11.4s, %20.s[2]     \n"
									"fmla   v7.4s, v11.4s, %23.s[2]     \n"
									"fmla   v12.4s, v11.4s, %19.s[2]    \n"
									"fmla   v13.4s, v11.4s, %22.s[2]    \n"

									"prfm   pldl1keep, [%8, #256]       \n"
									"ld1    {v14.4s, v15.4s}, [%8]      \n"// r3
									"add    %8, %8, #16                 \n"

									"ext    v10.16b, v8.16b, v9.16b, #4 \n"

									"st1    {v6.4s}, [%1], #16          \n"
									"st1    {v7.4s}, [%2], #16          \n"

									"ext    v11.16b, v14.16b, v15.16b, #8 \n"

									"st1    {v12.4s}, [%3], #16         \n"
									"st1    {v13.4s}, [%4], #16         \n"

									"subs   %w0, %w0, #1                \n"
									"bne    0b                          \n"

									"sub    %5, %5, #16                 \n"
									"sub    %8, %8, #16                 \n"
									: "=r"(nn),         // %0
									"=r"(outptr0),    // %1
									"=r"(outptr1),    // %2
									"=r"(outptr0n),   // %3
									"=r"(outptr1n),   // %4
									"=r"(r0),         // %5
									"=r"(r1),         // %6
									"=r"(r2),         // %7
									"=r"(r3)          // %8
									: "0"(nn),
									"1"(outptr0),
									"2"(outptr1),
									"3"(outptr0n),
									"4"(outptr1n),
									"5"(r0),
									"6"(r1),
									"7"(r2),
									"8"(r3),
									"w"(_k00),      // %18
									"w"(_k03),      // %19
									"w"(_k06),      // %20
									"w"(_k10),      // %21
									"w"(_k13),      // %22
									"w"(_k16)       // %23
									: "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
									);
							}
#else
							if (nn > 0)
							{
								asm volatile(

									"pld        [%5, #192]          \n"
									"vld1.f32   {d16-d18}, [%5 :64] \n"// r0
									"add        %5, #16             \n"

									"pld        [%8, #192]          \n"
									"vld1.f32   {d28-d30}, [%8]     \n"// r3
									"add        %8, #16             \n"

									"vext.32    q10, q8, q9, #1     \n"
									"vext.32    q11, q14, q15, #2   \n"

									"0:                             \n"

									"pld        [%1, #128]          \n"
									"vld1.f32   {d12-d13}, [%1 :64] \n"// _sum0

									"pld        [%2, #128]          \n"
									"vld1.f32   {d14-d15}, [%2 :64] \n"// _sum1

									"vmla.f32   q6, q8, %e18[0]     \n"
									"vmla.f32   q7, q8, %e21[0]     \n"

									"pld        [%3, #128]          \n"
									"vld1.f32   {d24-d25}, [%3]     \n"// _sum0n

									"pld        [%4, #128]          \n"
									"vld1.f32   {d26-d27}, [%4]     \n"// _sum1n

									"vmla.f32   q12, q14, %e20[0]   \n"
									"vmla.f32   q13, q14, %e23[0]   \n"

									"vext.32    q8, q8, q9, #2      \n"
									"vext.32    q9, q14, q15, #1    \n"

									"vmla.f32   q6, q10, %e18[1]    \n"
									"vmla.f32   q7, q10, %e21[1]    \n"
									"vmla.f32   q12, q11, %f20[0]   \n"
									"vmla.f32   q13, q11, %f23[0]   \n"

									"pld        [%6, #192]          \n"
									"vld1.f32   {d28-d30}, [%6]     \n"// r1
									"add        %6, #16             \n"

									"vmla.f32   q6, q8, %f18[0]     \n"
									"vmla.f32   q7, q8, %f21[0]     \n"
									"vmla.f32   q12, q9, %e20[1]    \n"
									"vmla.f32   q13, q9, %e23[1]    \n"

									"vext.32    q10, q14, q15, #1   \n"

									"vmla.f32   q6, q14, %e19[0]    \n"
									"vmla.f32   q7, q14, %e22[0]    \n"
									"vmla.f32   q12, q14, %e18[0]   \n"
									"vmla.f32   q13, q14, %e21[0]   \n"

									"vext.32    q11, q14, q15, #2   \n"

									"vmla.f32   q6, q10, %e19[1]    \n"
									"vmla.f32   q7, q10, %e22[1]    \n"
									"vmla.f32   q12, q10, %e18[1]   \n"
									"vmla.f32   q13, q10, %e21[1]   \n"

									"pld        [%7, #192]          \n"
									"vld1.f32   {d16-d18}, [%7 :64] \n"// r2
									"add        %7, #16             \n"

									"vmla.f32   q6, q11, %f19[0]    \n"
									"vmla.f32   q7, q11, %f22[0]    \n"
									"vmla.f32   q12, q11, %f18[0]   \n"
									"vmla.f32   q13, q11, %f21[0]   \n"

									"vext.32    q10, q8, q9, #1     \n"

									"vmla.f32   q6, q8, %e20[0]     \n"
									"vmla.f32   q7, q8, %e23[0]     \n"
									"vmla.f32   q12, q8, %e19[0]    \n"
									"vmla.f32   q13, q8, %e22[0]    \n"

									"vext.32    q11, q8, q9, #2     \n"

									"vmla.f32   q6, q10, %e20[1]    \n"
									"vmla.f32   q7, q10, %e23[1]    \n"
									"vmla.f32   q12, q10, %e19[1]   \n"
									"vmla.f32   q13, q10, %e22[1]   \n"

									"pld        [%5, #192]          \n"
									"vld1.f32   {d16-d18}, [%5 :64] \n"// r0
									"add        %5, #16             \n"

									"vmla.f32   q6, q11, %f20[0]    \n"
									"vmla.f32   q7, q11, %f23[0]    \n"
									"vmla.f32   q12, q11, %f19[0]   \n"
									"vmla.f32   q13, q11, %f22[0]   \n"

									"pld        [%8, #192]          \n"
									"vld1.f32   {d28-d30}, [%8]     \n"// r3
									"add        %8, #16             \n"

									"vext.32    q10, q8, q9, #1     \n"

									"vst1.f32   {d12-d13}, [%1 : 64]!\n"
									"vst1.f32   {d14-d15}, [%2 : 64]!\n"

									"vext.32    q11, q14, q15, #2   \n"

									"vst1.f32   {d24-d25}, [%3]!    \n"
									"vst1.f32   {d26-d27}, [%4]!    \n"

									"subs       %0, #1              \n"
									"bne        0b                  \n"

									"sub        %5, #16             \n"
									"sub        %8, #16             \n"
									: "=r"(nn),         // %0
									"=r"(outptr0),    // %1
									"=r"(outptr1),    // %2
									"=r"(outptr0n),   // %3
									"=r"(outptr1n),   // %4
									"=r"(r0),         // %5
									"=r"(r1),         // %6
									"=r"(r2),         // %7
									"=r"(r3)          // %8
									: "0"(nn),
									"1"(outptr0),
									"2"(outptr1),
									"3"(outptr0n),
									"4"(outptr1n),
									"5"(r0),
									"6"(r1),
									"7"(r2),
									"8"(r3),
									"w"(_k00),      // %18
									"w"(_k03),      // %19
									"w"(_k06),      // %20
									"w"(_k10),      // %21
									"w"(_k13),      // %22
									"w"(_k16)       // %23
									: "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
									);
							}
#endif // __aarch64__
#endif // __ARM_NEON
							for (; remain > 0; remain--)
							{
#if __ARM_NEON
								float32x4_t _r00 = vld1q_f32(r0);
								float32x4_t _r10 = vld1q_f32(r1);
								float32x4_t _r20 = vld1q_f32(r2);
								float32x4_t _r30 = vld1q_f32(r3);

								float32x4_t _sum0 = vmulq_f32(_r00, _k00);
								float32x4_t _sum1 = vmulq_f32(_r00, _k10);
								_sum0 = vmlaq_f32(_sum0, _r10, _k03);
								_sum1 = vmlaq_f32(_sum1, _r10, _k13);
								_sum0 = vmlaq_f32(_sum0, _r20, _k06);
								_sum1 = vmlaq_f32(_sum1, _r20, _k16);

								float32x4_t _sum0n = vmulq_f32(_r10, _k00);
								float32x4_t _sum1n = vmulq_f32(_r10, _k10);
								_sum0n = vmlaq_f32(_sum0n, _r20, _k03);
								_sum1n = vmlaq_f32(_sum1n, _r20, _k13);
								_sum0n = vmlaq_f32(_sum0n, _r30, _k06);
								_sum1n = vmlaq_f32(_sum1n, _r30, _k16);

								_sum0 = vsetq_lane_f32(*outptr0, _sum0, 3);
								_sum1 = vsetq_lane_f32(*outptr1, _sum1, 3);
								_sum0n = vsetq_lane_f32(*outptr0n, _sum0n, 3);
								_sum1n = vsetq_lane_f32(*outptr1n, _sum1n, 3);
#if __aarch64__
								*outptr0 = vaddvq_f32(_sum0);
								*outptr1 = vaddvq_f32(_sum1);
								*outptr0n = vaddvq_f32(_sum0n);
								*outptr1n = vaddvq_f32(_sum1n);
#else
								float32x2_t _ss0 = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
								float32x2_t _ss1 = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
								float32x2_t _ss0n = vadd_f32(vget_low_f32(_sum0n), vget_high_f32(_sum0n));
								float32x2_t _ss1n = vadd_f32(vget_low_f32(_sum1n), vget_high_f32(_sum1n));

								float32x2_t _ss01 = vpadd_f32(_ss0, _ss1);
								float32x2_t _ss01n = vpadd_f32(_ss0n, _ss1n);

								*outptr0 = vget_lane_f32(_ss01, 0);
								*outptr1 = vget_lane_f32(_ss01, 1);
								*outptr0n = vget_lane_f32(_ss01n, 0);
								*outptr1n = vget_lane_f32(_ss01n, 1);
#endif // __aarch64__
#else
								*outptr0 += mul_add_3x3_native(r0, r1, r2, k0, k0 + 3, k0 + 6, 0);
								*outptr1 += mul_add_3x3_native(r0, r1, r2, k1, k1 + 3, k1 + 6, 0);
								*outptr0n += mul_add_3x3_native(r1, r2, r3, k0, k0 + 3, k0 + 6, 0);
								*outptr1n += mul_add_3x3_native(r1, r2, r3, k1, k1 + 3, k1 + 6, 0);

#endif // __ARM_NEON
								r0++;
								r1++;
								r2++;
								r3++;
								outptr0++;
								outptr1++;
								outptr0n++;
								outptr1n++;
							}

							r0 += 2 + w;
							r1 += 2 + w;
							r2 += 2 + w;
							r3 += 2 + w;

							outptr0 += outw;
							outptr1 += outw;
							outptr0n += outw;
							outptr1n += outw;
						}

						for (; i < outh; i++)
						{
#if __ARM_NEON
							int nn = outw >> 2;
							int remain = outw & 3;
#else
							int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
							if (nn > 0)
							{
								asm volatile(
									"0:                                 \n"

									"prfm   pldl1keep, [%3, #256]       \n"
									"ld1    {v8.4s, v9.4s}, [%3]        \n"// r0
									"add    %3, %3, #16                 \n"

									"prfm   pldl1keep, [%1, #128]       \n"
									"ld1    {v6.4s}, [%1]               \n"// _sum0

									"prfm   pldl1keep, [%2, #128]       \n"
									"ld1    {v7.4s}, [%2]               \n"// _sum1

									"fmul   v14.4s, v8.4s, %12.s[0]     \n"
									"fmul   v15.4s, v8.4s, %15.s[0]     \n"

									"ext    v10.16b, v8.16b, v9.16b, #4 \n"
									"ext    v11.16b, v8.16b, v9.16b, #8 \n"

									"fmla   v6.4s, v10.4s, %12.s[1]     \n"
									"fmla   v7.4s, v10.4s, %15.s[1]     \n"

									"prfm   pldl1keep, [%4, #256]       \n"
									"ld1    {v8.4s, v9.4s}, [%4]        \n"// r1
									"add    %4, %4, #16                 \n"

									"fmla   v14.4s, v11.4s, %12.s[2]    \n"
									"fmla   v15.4s, v11.4s, %15.s[2]    \n"

									"fmla   v6.4s, v8.4s, %13.s[0]      \n"
									"fmla   v7.4s, v8.4s, %16.s[0]      \n"

									"ext    v10.16b, v8.16b, v9.16b, #4 \n"
									"ext    v11.16b, v8.16b, v9.16b, #8 \n"

									"fmla   v14.4s, v10.4s, %13.s[1]    \n"
									"fmla   v15.4s, v10.4s, %16.s[1]    \n"

									"prfm   pldl1keep, [%5, #256]       \n"
									"ld1    {v8.4s, v9.4s}, [%5]        \n"// r2
									"add    %5, %5, #16                 \n"

									"fmla   v6.4s, v11.4s, %13.s[2]     \n"
									"fmla   v7.4s, v11.4s, %16.s[2]     \n"

									"fmla   v14.4s, v8.4s, %14.s[0]     \n"
									"fmla   v15.4s, v8.4s, %17.s[0]     \n"

									"ext    v10.16b, v8.16b, v9.16b, #4 \n"
									"ext    v11.16b, v8.16b, v9.16b, #8 \n"

									"fmla   v6.4s, v10.4s, %14.s[1]     \n"
									"fmla   v7.4s, v10.4s, %17.s[1]     \n"

									"fmla   v14.4s, v11.4s, %14.s[2]    \n"
									"fmla   v15.4s, v11.4s, %17.s[2]    \n"

									"fadd   v6.4s, v6.4s, v14.4s        \n"
									"fadd   v7.4s, v7.4s, v15.4s        \n"

									"st1    {v6.4s}, [%1], #16          \n"
									"st1    {v7.4s}, [%2], #16          \n"

									"subs   %w0, %w0, #1                \n"
									"bne    0b                          \n"

									: "=r"(nn),         // %0
									"=r"(outptr0),    // %1
									"=r"(outptr1),    // %2
									"=r"(r0),         // %3
									"=r"(r1),         // %4
									"=r"(r2)          // %5
									: "0"(nn),
									"1"(outptr0),
									"2"(outptr1),
									"3"(r0),
									"4"(r1),
									"5"(r2),
									"w"(_k00),      // %12
									"w"(_k03),      // %13
									"w"(_k06),      // %14
									"w"(_k10),      // %15
									"w"(_k13),      // %16
									"w"(_k16)       // %17
									: "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
									);
							}
#else
							if (nn > 0)
							{
								asm volatile(
									"0:                             \n"

									"pld        [%3, #192]          \n"
									"vld1.f32   {d16-d18}, [%3]     \n"// r0
									"add        %3, #16             \n"

									"pld        [%1, #128]          \n"
									"vld1.f32   {d12-d13}, [%1]     \n"// _sum0

									"pld        [%2, #128]          \n"
									"vld1.f32   {d14-d15}, [%2]     \n"// _sum1

									"vmul.f32   q14, q8, %e12[0]    \n"
									"vmul.f32   q15, q8, %e15[0]    \n"

									"vext.32    q10, q8, q9, #1     \n"
									"vext.32    q11, q8, q9, #2     \n"

									"vmla.f32   q6, q10, %e12[1]    \n"
									"vmla.f32   q7, q10, %e15[1]    \n"

									"pld        [%4, #192]          \n"
									"vld1.f32   {d16-d18}, [%4]     \n"// r1
									"add        %4, #16             \n"

									"vmla.f32   q14, q11, %f12[0]   \n"
									"vmla.f32   q15, q11, %f15[0]   \n"

									"vmla.f32   q6, q8, %e13[0]     \n"
									"vmla.f32   q7, q8, %e16[0]     \n"

									"vext.32    q10, q8, q9, #1     \n"
									"vext.32    q11, q8, q9, #2     \n"

									"vmla.f32   q14, q10, %e13[1]   \n"
									"vmla.f32   q15, q10, %e16[1]   \n"

									"pld        [%5, #192]          \n"
									"vld1.f32   {d16-d18}, [%5]     \n"// r2
									"add        %5, #16             \n"

									"vmla.f32   q6, q11, %f13[0]    \n"
									"vmla.f32   q7, q11, %f16[0]    \n"

									"vmla.f32   q14, q8, %e14[0]    \n"
									"vmla.f32   q15, q8, %e17[0]    \n"

									"vext.32    q10, q8, q9, #1     \n"
									"vext.32    q11, q8, q9, #2     \n"

									"vmla.f32   q6, q10, %e14[1]    \n"
									"vmla.f32   q7, q10, %e17[1]    \n"

									"vmla.f32   q14, q11, %f14[0]   \n"
									"vmla.f32   q15, q11, %f17[0]   \n"

									"vadd.f32   q6, q6, q14         \n"
									"vadd.f32   q7, q7, q15         \n"

									"vst1.f32   {d12-d13}, [%1]!    \n"

									"vst1.f32   {d14-d15}, [%2]!    \n"

									"subs       %0, #1              \n"
									"bne        0b                  \n"

									: "=r"(nn),         // %0
									"=r"(outptr0),    // %1
									"=r"(outptr1),    // %2
									"=r"(r0),         // %3
									"=r"(r1),         // %4
									"=r"(r2)          // %5
									: "0"(nn),
									"1"(outptr0),
									"2"(outptr1),
									"3"(r0),
									"4"(r1),
									"5"(r2),
									"w"(_k00),      // %12
									"w"(_k03),      // %13
									"w"(_k06),      // %14
									"w"(_k10),      // %15
									"w"(_k13),      // %16
									"w"(_k16)       // %17
									: "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
									);
							}
#endif // __aarch64__
#endif // __ARM_NEON
							for (; remain > 0; remain--)
							{
#if __ARM_NEON
								float32x4_t _r00 = vld1q_f32(r0);
								float32x4_t _r10 = vld1q_f32(r1);
								float32x4_t _r20 = vld1q_f32(r2);

								float32x4_t _sum0 = vmulq_f32(_r00, _k00);
								float32x4_t _sum1 = vmulq_f32(_r00, _k10);
								_sum0 = vmlaq_f32(_sum0, _r10, _k03);
								_sum1 = vmlaq_f32(_sum1, _r10, _k13);
								_sum0 = vmlaq_f32(_sum0, _r20, _k06);
								_sum1 = vmlaq_f32(_sum1, _r20, _k16);

								_sum0 = vsetq_lane_f32(*outptr0, _sum0, 3);
								_sum1 = vsetq_lane_f32(*outptr1, _sum1, 3);
#if __aarch64__
								*outptr0 = vaddvq_f32(_sum0);
								*outptr1 = vaddvq_f32(_sum1);
#else
								float32x2_t _ss0 = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
								float32x2_t _ss1 = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
								float32x2_t _ss01 = vpadd_f32(_ss0, _ss1);

								*outptr0 = vget_lane_f32(_ss01, 0);
								*outptr1 = vget_lane_f32(_ss01, 1);
#endif // __aarch64__
#else
								*outptr0 += mul_add_3x3_native(r0, r1, r2, k0, k0 + 3, k0 + 6, 0);
								*outptr1 += mul_add_3x3_native(r0, r1, r2, k1, k1 + 3, k1 + 6, 0);

#endif // __ARM_NEON
								r0++;
								r1++;
								r2++;
								outptr0++;
								outptr1++;
							}

							r0 += 2;
							r1 += 2;
							r2 += 2;
						}

						k0 += 9;
						k1 += 9;
					}
				}

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
				for (int p = remain_outch_start; p < outch; p++)
				{
					float *out = top_data + (p)* top_cstep;

					const float bias0 = this->bias_term_ ? bias[p] : 0.f;

					fill(out, top_cstep, bias0);

					const float* kernel0 = kernel + p * inch * 9;

					for (int q = 0; q < inch; q++)
					{
						float* outptr = out;
						float* outptr2 = outptr + outw;

						const float* img0 = bottom_data + (q)* bottom_cstep;

						const float* r0 = img0;
						const float* r1 = img0 + w;
						const float* r2 = img0 + w * 2;
						const float* r3 = img0 + w * 3;

#if __ARM_NEON
						float32x4_t _k0123 = vld1q_f32(kernel0);
						float32x4_t _k3456 = vld1q_f32(kernel0 + 3);
						float32x4_t _k6789 = vld1q_f32(kernel0 + 6);
#else
						const float* k0 = kernel0;
						const float* k1 = kernel0 + 3;
						const float* k2 = kernel0 + 6;

#endif // __ARM_NEON

						int i = 0;

						for (; i + 1 < outh; i += 2)
						{

#if __ARM_NEON
							int nn = outw >> 2;
							int remain = outw & 3;
#else
							int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
							if (nn > 0)
							{
								asm volatile(
									"prfm   pldl1keep, [%3, #256]       \n"
									"ld1    {v9.4s, v10.4s}, [%3]       \n"// r0
									"add    %3, %3, #16                 \n"

									"ext    v11.16b, v9.16b, v10.16b, #4 \n"
									"ext    v12.16b, v9.16b, v10.16b, #8 \n"

									"0:                                 \n"

									"prfm   pldl1keep, [%1, #128]       \n"
									"ld1    {v7.4s}, [%1]               \n"// _sum

									"fmla   v7.4s, v9.4s, %14.s[0]      \n"
									"fmul   v6.4s, v11.4s, %14.s[1]     \n"
									"fmul   v13.4s, v12.4s, %14.s[2]    \n"

									"prfm   pldl1keep, [%4, #256]       \n"
									"ld1    {v9.4s, v10.4s}, [%4]       \n"// r1
									"add    %4, %4, #16                 \n"

									"fmla   v7.4s, v9.4s, %15.s[0]      \n"

									"ext    v11.16b, v9.16b, v10.16b, #4 \n"
									"ext    v12.16b, v9.16b, v10.16b, #8 \n"

									"fmla   v6.4s, v11.4s, %15.s[1]     \n"
									"fmla   v13.4s, v12.4s, %15.s[2]    \n"

									"prfm   pldl1keep, [%2, #128]       \n"
									"ld1    {v8.4s}, [%2]               \n"// _sum2

									"fmla   v8.4s, v9.4s, %14.s[0]      \n"
									"fmul   v14.4s, v11.4s, %14.s[1]    \n"
									"fmul   v15.4s, v12.4s, %14.s[2]    \n"

									"prfm   pldl1keep, [%5, #256]       \n"
									"ld1    {v9.4s, v10.4s}, [%5]       \n"// r2
									"add    %5, %5, #16                 \n"

									"fmla   v7.4s, v9.4s, %16.s[0]      \n"

									"ext    v11.16b, v9.16b, v10.16b, #4 \n"
									"ext    v12.16b, v9.16b, v10.16b, #8 \n"

									"fmla   v6.4s, v11.4s, %16.s[1]     \n"
									"fmla   v13.4s, v12.4s, %16.s[2]    \n"

									"fmla   v8.4s, v9.4s, %15.s[0]      \n"
									"fmla   v14.4s, v11.4s, %15.s[1]    \n"
									"fmla   v15.4s, v12.4s, %15.s[2]    \n"

									"prfm   pldl1keep, [%6, #256]       \n"
									"ld1    {v9.4s, v10.4s}, [%6]       \n"// r3
									"add    %6, %6, #16                 \n"

									"fmla   v8.4s, v9.4s, %16.s[0]      \n"

									"ext    v11.16b, v9.16b, v10.16b, #4 \n"
									"ext    v12.16b, v9.16b, v10.16b, #8 \n"

									"fmla   v14.4s, v11.4s, %16.s[1]    \n"
									"fmla   v15.4s, v12.4s, %16.s[2]    \n"

									"fadd   v7.4s, v7.4s, v6.4s         \n"

									"prfm   pldl1keep, [%3, #256]       \n"
									"ld1    {v9.4s, v10.4s}, [%3]       \n"// r0

									"fadd   v8.4s, v8.4s, v14.4s        \n"
									"fadd   v7.4s, v7.4s, v13.4s        \n"
									"fadd   v8.4s, v8.4s, v15.4s        \n"

									"ext    v11.16b, v9.16b, v10.16b, #4 \n"
									"ext    v12.16b, v9.16b, v10.16b, #8 \n"

									"add    %3, %3, #16                 \n"

									"st1    {v7.4s}, [%1], #16          \n"
									"st1    {v8.4s}, [%2], #16          \n"

									"subs   %w0, %w0, #1                \n"
									"bne    0b                          \n"

									"sub    %3, %3, #16                 \n"
									: "=r"(nn),         // %0
									"=r"(outptr),     // %1
									"=r"(outptr2),    // %2
									"=r"(r0),         // %3
									"=r"(r1),         // %4
									"=r"(r2),         // %5
									"=r"(r3)          // %6
									: "0"(nn),
									"1"(outptr),
									"2"(outptr2),
									"3"(r0),
									"4"(r1),
									"5"(r2),
									"6"(r3),
									"w"(_k0123),      // %14
									"w"(_k3456),      // %15
									"w"(_k6789)       // %16
									: "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
									);
							}
#else
							if (nn > 0)
							{
								asm volatile(
									"pld        [%3, #192]          \n"
									"vld1.f32   {d18-d20}, [%3 :64] \n"// r0
									"add        %3, #16             \n"

									"vext.32    q11, q9, q10, #1    \n"
									"vext.32    q12, q9, q10, #2    \n"

									"0:                             \n"

									"pld        [%1, #128]          \n"
									"vld1.f32   {d14-d15}, [%1 :64] \n"// _sum

									"vmla.f32   q7, q9, %e14[0]     \n"
									"vmul.f32   q6, q11, %e14[1]    \n"
									"vmul.f32   q13, q12, %f14[0]   \n"

									"pld        [%4, #192]          \n"
									"vld1.f32   {d18-d20}, [%4]     \n"// r1
									"add        %4, #16             \n"

									"vmla.f32   q7, q9, %e15[0]     \n"

									"vext.32    q11, q9, q10, #1    \n"
									"vext.32    q12, q9, q10, #2    \n"

									"vmla.f32   q6, q11, %e15[1]    \n"
									"vmla.f32   q13, q12, %f15[0]   \n"

									"pld        [%2, #128]          \n"
									"vld1.f32   {d16-d17}, [%2]     \n"// _sum2

									"vmla.f32   q8, q9, %e14[0]     \n"
									"vmul.f32   q14, q11, %e14[1]   \n"
									"vmul.f32   q15, q12, %f14[0]   \n"

									"pld        [%5, #192]          \n"
									"vld1.f32   {d18-d20}, [%5 :64] \n"// r2
									"add        %5, #16             \n"

									"vmla.f32   q7, q9, %e16[0]     \n"

									"vext.32    q11, q9, q10, #1    \n"
									"vext.32    q12, q9, q10, #2    \n"

									"vmla.f32   q6, q11, %e16[1]    \n"
									"vmla.f32   q13, q12, %f16[0]   \n"

									"vmla.f32   q8, q9, %e15[0]     \n"
									"vmla.f32   q14, q11, %e15[1]   \n"
									"vmla.f32   q15, q12, %f15[0]   \n"

									"pld        [%6, #192]          \n"
									"vld1.f32   {d18-d20}, [%6]     \n"// r3
									"add        %6, #16             \n"

									"vmla.f32   q8, q9, %e16[0]     \n"

									"vext.32    q11, q9, q10, #1    \n"
									"vext.32    q12, q9, q10, #2    \n"

									"vmla.f32   q14, q11, %e16[1]   \n"
									"vmla.f32   q15, q12, %f16[0]   \n"

									"vadd.f32   q7, q7, q6          \n"

									"pld        [%3, #192]          \n"
									"vld1.f32   {d18-d20}, [%3 :64] \n"// r0

									"vadd.f32   q8, q8, q14         \n"
									"vadd.f32   q7, q7, q13         \n"
									"vadd.f32   q8, q8, q15         \n"

									"vext.32    q11, q9, q10, #1    \n"
									"vext.32    q12, q9, q10, #2    \n"

									"add        %3, #16             \n"

									"vst1.f32   {d14-d15}, [%1]!    \n"
									"vst1.f32   {d16-d17}, [%2]!    \n"

									"subs       %0, #1              \n"
									"bne        0b                  \n"

									"sub        %3, #16             \n"
									: "=r"(nn),         // %0
									"=r"(outptr),     // %1
									"=r"(outptr2),    // %2
									"=r"(r0),         // %3
									"=r"(r1),         // %4
									"=r"(r2),         // %5
									"=r"(r3)          // %6
									: "0"(nn),
									"1"(outptr),
									"2"(outptr2),
									"3"(r0),
									"4"(r1),
									"5"(r2),
									"6"(r3),
									"w"(_k0123),      // %14
									"w"(_k3456),      // %15
									"w"(_k6789)       // %16
									: "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
									);
							}
#endif // __aarch64__
#endif // __ARM_NEON
							for (; remain > 0; remain--)
							{
#if __ARM_NEON
								float32x4_t _r00 = vld1q_f32(r0);
								float32x4_t _r10 = vld1q_f32(r1);
								float32x4_t _r20 = vld1q_f32(r2);
								float32x4_t _r30 = vld1q_f32(r3);

								float32x4_t _sum = vmulq_f32(_r00, _k0123);
								_sum = vmlaq_f32(_sum, _r10, _k3456);
								_sum = vmlaq_f32(_sum, _r20, _k6789);

								float32x4_t _sum2 = vmulq_f32(_r10, _k0123);
								_sum2 = vmlaq_f32(_sum2, _r20, _k3456);
								_sum2 = vmlaq_f32(_sum2, _r30, _k6789);

								_sum = vsetq_lane_f32(*outptr, _sum, 3);
								_sum2 = vsetq_lane_f32(*outptr2, _sum2, 3);

#if __aarch64__
								*outptr = vaddvq_f32(_sum);
								*outptr2 = vaddvq_f32(_sum2);
#else
								float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
								float32x2_t _ss2 = vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));

								float32x2_t _sss2 = vpadd_f32(_ss, _ss2);

								*outptr = vget_lane_f32(_sss2, 0);
								*outptr2 = vget_lane_f32(_sss2, 1);
#endif // __aarch64__
#else
								*outptr += mul_add_3x3_native(r0, r1, r2, k0, k1, k2, 0);
								*outptr2 += mul_add_3x3_native(r1, r2, r3, k0, k1, k2, 0);
#endif
								r0++;
								r1++;
								r2++;
								r3++;
								outptr++;
								outptr2++;
							}

							r0 += 2 + w;
							r1 += 2 + w;
							r2 += 2 + w;
							r3 += 2 + w;

							outptr += outw;
							outptr2 += outw;
						}

						for (; i < outh; i++)
						{

#if __ARM_NEON
							int nn = outw >> 2;
							int remain = outw & 3;
#else
							int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
							if (nn > 0)
							{
								asm volatile(
									"prfm   pldl1keep, [%2, #256]       \n"
									"ld1    {v8.4s, v9.4s}, [%2]        \n"// r0
									"add    %2, %2, #16                 \n"

									"ext    v10.16b, v8.16b, v9.16b, #4 \n"
									"ext    v11.16b, v8.16b, v9.16b, #8 \n"

									"0:                                 \n"

									"prfm   pldl1keep, [%1, #128]       \n"
									"ld1    {v7.4s}, [%1]               \n"// _sum

									"fmla   v7.4s, v8.4s, %10.s[0]      \n"
									"fmul   v13.4s, v10.4s, %10.s[1]    \n"
									"fmul   v14.4s, v11.4s, %10.s[2]    \n"

									"prfm   pldl1keep, [%3, #256]       \n"
									"ld1    {v8.4s, v9.4s}, [%3]        \n"// r1
									"add    %3, %3, #16                 \n"

									"fmla   v7.4s, v8.4s, %11.s[0]      \n"

									"ext    v10.16b, v8.16b, v9.16b, #4 \n"
									"ext    v11.16b, v8.16b, v9.16b, #8 \n"

									"fmla   v13.4s, v10.4s, %11.s[1]    \n"
									"fmla   v14.4s, v11.4s, %11.s[2]    \n"

									"prfm   pldl1keep, [%4, #256]       \n"
									"ld1    {v8.4s, v9.4s}, [%4]        \n"// r2
									"add    %4, %4, #16                 \n"

									"fmla   v7.4s, v8.4s, %12.s[0]      \n"

									"ext    v10.16b, v8.16b, v9.16b, #4 \n"
									"ext    v11.16b, v8.16b, v9.16b, #8 \n"

									"fmla   v13.4s, v10.4s, %12.s[1]    \n"
									"fmla   v14.4s, v11.4s, %12.s[2]    \n"

									"prfm   pldl1keep, [%2, #256]       \n"
									"ld1    {v8.4s, v9.4s}, [%2]        \n"// r0
									"add    %2, %2, #16                 \n"

									"fadd   v7.4s, v7.4s, v13.4s        \n"
									"fadd   v7.4s, v7.4s, v14.4s        \n"

									"ext    v10.16b, v8.16b, v9.16b, #4 \n"
									"ext    v11.16b, v8.16b, v9.16b, #8 \n"

									"st1    {v7.4s}, [%1], #16          \n"

									"subs   %w0, %w0, #1                \n"
									"bne    0b                          \n"

									"sub    %2, %2, #16                 \n"
									: "=r"(nn),         // %0
									"=r"(outptr),     // %1
									"=r"(r0),         // %2
									"=r"(r1),         // %3
									"=r"(r2)          // %4
									: "0"(nn),
									"1"(outptr),
									"2"(r0),
									"3"(r1),
									"4"(r2),
									"w"(_k0123),      // %10
									"w"(_k3456),      // %11
									"w"(_k6789)       // %12
									: "cc", "memory", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
									);
							}
#else
							if (nn > 0)
							{
								asm volatile(
									"pld        [%2, #192]          \n"
									"vld1.f32   {d16-d18}, [%2]     \n"// r0
									"add        %2, #16             \n"

									"vext.32    q10, q8, q9, #1     \n"
									"vext.32    q11, q8, q9, #2     \n"

									"0:                             \n"

									"pld        [%1, #128]          \n"
									"vld1.f32   {d14-d15}, [%1]     \n"// _sum

									"vmla.f32   q7, q8, %e10[0]     \n"
									"vmul.f32   q13, q10, %e10[1]   \n"
									"vmul.f32   q14, q11, %f10[0]   \n"

									"pld        [%3, #192]          \n"
									"vld1.f32   {d16-d18}, [%3]     \n"// r1
									"add        %3, #16             \n"

									"vmla.f32   q7, q8, %e11[0]     \n"

									"vext.32    q10, q8, q9, #1     \n"
									"vext.32    q11, q8, q9, #2     \n"

									"vmla.f32   q13, q10, %e11[1]   \n"
									"vmla.f32   q14, q11, %f11[0]   \n"

									"pld        [%4, #192]          \n"
									"vld1.f32   {d16-d18}, [%4]     \n"// r2
									"add        %4, #16             \n"

									"vmla.f32   q7, q8, %e12[0]     \n"

									"vext.32    q10, q8, q9, #1     \n"
									"vext.32    q11, q8, q9, #2     \n"

									"vmla.f32   q13, q10, %e12[1]   \n"
									"vmla.f32   q14, q11, %f12[0]   \n"

									"pld        [%2, #192]          \n"
									"vld1.f32   {d16-d18}, [%2]     \n"// r0
									"add        %2, #16             \n"

									"vadd.f32   q7, q7, q13         \n"
									"vadd.f32   q7, q7, q14         \n"

									"vext.32    q10, q8, q9, #1     \n"
									"vext.32    q11, q8, q9, #2     \n"

									"vst1.f32   {d14-d15}, [%1]!    \n"

									"subs       %0, #1              \n"
									"bne        0b                  \n"

									"sub        %2, #16             \n"
									: "=r"(nn),         // %0
									"=r"(outptr),     // %1
									"=r"(r0),         // %2
									"=r"(r1),         // %3
									"=r"(r2)          // %4
									: "0"(nn),
									"1"(outptr),
									"2"(r0),
									"3"(r1),
									"4"(r2),
									"w"(_k0123),      // %10
									"w"(_k3456),      // %11
									"w"(_k6789)       // %12
									: "cc", "memory", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
									);
							}
#endif // __aarch64__
#endif // __ARM_NEON
							for (; remain > 0; remain--)
							{
#if __ARM_NEON
								float32x4_t _r00 = vld1q_f32(r0);
								float32x4_t _r10 = vld1q_f32(r1);
								float32x4_t _r20 = vld1q_f32(r2);

								float32x4_t _sum = vmulq_f32(_r00, _k0123);
								_sum = vmlaq_f32(_sum, _r10, _k3456);
								_sum = vmlaq_f32(_sum, _r20, _k6789);

								_sum = vsetq_lane_f32(*outptr, _sum, 3);

#if __aarch64__
								*outptr = vaddvq_f32(_sum);
#else
								float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
								_ss = vpadd_f32(_ss, _ss);

								*outptr = vget_lane_f32(_ss, 0);
#endif // __aarch64__
#else
								*outptr += mul_add_3x3_native(r0, r1, r2, k0, k1, k2, 0);
#endif
								r0++;
								r1++;
								r2++;
								outptr++;
							}

							r0 += 2;
							r1 += 2;
							r2 += 2;
						}

						kernel0 += 9;
					}
				}
			}
		}

		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv3x3s2_packed_neon(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top)
		{
			const float *kernel_data = kernel_tm_->cpu_data();
			int kernel_cstep = kernel_tm_->width() * kernel_tm_->height();

			int num = bottom->num();
			int w = bottom->width();
			int h = bottom->height();
			int inch = bottom->channels();
			int bottom_cstep = w * h;

			int outw = top->width();
			int outh = top->height();
			int outch = top->channels();
			int top_cstep = outw * outh;

			const int tailstep = w - 2 * outw + w;

			const float* bias = nullptr;
			if (this->bias_term_)
				bias = this->weights_f32_[1]->cpu_data();

			for (int num_i = 0; num_i < num; num_i++)
			{
				const float *bottom_data = bottom->cpu_data() + num_i * inch * bottom_cstep;
				float * top_data = top->mutable_cpu_data() + num_i * outch * top_cstep;

				int nn_outch = outch >> 3;
				int remain_outch_start = nn_outch << 3;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
				for (int pp = 0; pp < nn_outch; pp++)
				{
					int p = pp * 8;

					float *out0 = top_data + (p + 0) * top_cstep;
					float *out1 = top_data + (p + 1) * top_cstep;
					float *out2 = top_data + (p + 2) * top_cstep;
					float *out3 = top_data + (p + 3) * top_cstep;
					float *out4 = top_data + (p + 4) * top_cstep;
					float *out5 = top_data + (p + 5) * top_cstep;
					float *out6 = top_data + (p + 6) * top_cstep;
					float *out7 = top_data + (p + 7) * top_cstep;

					const float bias0 = this->bias_term_ ? bias[p + 0] : 0.f;
					const float bias1 = this->bias_term_ ? bias[p + 1] : 0.f;
					const float bias2 = this->bias_term_ ? bias[p + 2] : 0.f;
					const float bias3 = this->bias_term_ ? bias[p + 3] : 0.f;
					const float bias4 = this->bias_term_ ? bias[p + 4] : 0.f;
					const float bias5 = this->bias_term_ ? bias[p + 5] : 0.f;
					const float bias6 = this->bias_term_ ? bias[p + 6] : 0.f;
					const float bias7 = this->bias_term_ ? bias[p + 7] : 0.f;

					fill(out0, top_cstep, bias0);
					fill(out1, top_cstep, bias1);
					fill(out2, top_cstep, bias2);
					fill(out3, top_cstep, bias3);
					fill(out4, top_cstep, bias4);
					fill(out5, top_cstep, bias5);
					fill(out6, top_cstep, bias6);
					fill(out7, top_cstep, bias7);

					const float* ktmp = kernel_data + (p / 8) * kernel_cstep;

					for (int q = 0; q < inch; q++)
					{
						float* outptr0 = out0;
						float* outptr1 = out1;
						float* outptr2 = out2;
						float* outptr3 = out3;
						float* outptr4 = out4;
						float* outptr5 = out5;
						float* outptr6 = out6;
						float* outptr7 = out7;

						const float* img0 = bottom_data + (q)* bottom_cstep;

						const float* r0 = img0;
						const float* r1 = img0 + w;
						const float* r2 = img0 + w * 2;

						int i = 0;

						for (; i < outh; i++)
						{
#if __ARM_NEON
							int nn = outw >> 2;
							int remain = outw & 3;
#else
							int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
							if (nn > 0)
							{
								asm volatile(
									"0:                                 \n"

									"prfm   pldl1keep, [%1, #128]       \n"
									"ld1    {v8.4s}, [%1]               \n"
									"prfm   pldl1keep, [%2, #128]       \n"
									"ld1    {v9.4s}, [%2]               \n"

									"prfm   pldl1keep, [%3, #128]       \n"
									"ld1    {v10.4s}, [%3]              \n"
									"prfm   pldl1keep, [%4, #128]       \n"
									"ld1    {v11.4s}, [%4]              \n"

									///
									"prfm   pldl1keep, [%9, #256]       \n"
									"ld2    {v4.4s, v5.4s}, [%9], #32   \n"// v4=00 v5=01

									"ld1    {v0.4s, v1.4s}, [%12], #32  \n"

									"fmla   v8.4s, v4.4s, v0.s[0]       \n"
									"fmla   v9.4s, v4.4s, v0.s[1]       \n"

									"prfm   pldl1keep, [%5, #128]       \n"
									"ld1    {v12.4s}, [%5]              \n"
									"prfm   pldl1keep, [%6, #128]       \n"
									"ld1    {v13.4s}, [%6]              \n"

									"fmla   v10.4s, v4.4s, v0.s[2]      \n"
									"fmla   v11.4s, v4.4s, v0.s[3]      \n"

									"prfm   pldl1keep, [%7, #128]       \n"
									"ld1    {v14.4s}, [%7]              \n"
									"prfm   pldl1keep, [%8, #128]       \n"
									"ld1    {v15.4s}, [%8]              \n"

									"ld1    {v2.4s, v3.4s}, [%12], #32  \n"

									"fmla   v12.4s, v4.4s, v1.s[0]      \n"
									"fmla   v13.4s, v4.4s, v1.s[1]      \n"
									"fmla   v14.4s, v4.4s, v1.s[2]      \n"
									"fmla   v15.4s, v4.4s, v1.s[3]      \n"

									"prfm   pldl1keep, [%9, #256]       \n"
									"ld2    {v6.4s, v7.4s}, [%9]        \n"// v6

									"fmla   v8.4s, v5.4s, v2.s[0]       \n"
									"fmla   v9.4s, v5.4s, v2.s[1]       \n"
									"fmla   v10.4s, v5.4s, v2.s[2]      \n"
									"fmla   v11.4s, v5.4s, v2.s[3]      \n"

									"ext    v6.16b, v4.16b, v6.16b, #4  \n"// v6=02

									"ld1    {v0.4s, v1.4s}, [%12], #32  \n"

									"fmla   v12.4s, v5.4s, v3.s[0]      \n"
									"fmla   v13.4s, v5.4s, v3.s[1]      \n"
									"fmla   v14.4s, v5.4s, v3.s[2]      \n"
									"fmla   v15.4s, v5.4s, v3.s[3]      \n"

									///
									"prfm   pldl1keep, [%10, #256]      \n"
									"ld2    {v4.4s, v5.4s}, [%10], #32  \n"// v4=10 v5=11

									"fmla   v8.4s, v6.4s, v0.s[0]       \n"
									"fmla   v9.4s, v6.4s, v0.s[1]       \n"
									"fmla   v10.4s, v6.4s, v0.s[2]      \n"
									"fmla   v11.4s, v6.4s, v0.s[3]      \n"

									"ld1    {v2.4s, v3.4s}, [%12], #32  \n"

									"fmla   v12.4s, v6.4s, v1.s[0]      \n"
									"fmla   v13.4s, v6.4s, v1.s[1]      \n"
									"fmla   v14.4s, v6.4s, v1.s[2]      \n"
									"fmla   v15.4s, v6.4s, v1.s[3]      \n"

									"fmla   v8.4s, v4.4s, v2.s[0]       \n"
									"fmla   v9.4s, v4.4s, v2.s[1]       \n"
									"fmla   v10.4s, v4.4s, v2.s[2]      \n"
									"fmla   v11.4s, v4.4s, v2.s[3]      \n"

									"ld1    {v0.4s, v1.4s}, [%12], #32  \n"

									"fmla   v12.4s, v4.4s, v3.s[0]      \n"
									"fmla   v13.4s, v4.4s, v3.s[1]      \n"
									"fmla   v14.4s, v4.4s, v3.s[2]      \n"
									"fmla   v15.4s, v4.4s, v3.s[3]      \n"

									"prfm   pldl1keep, [%10, #256]      \n"
									"ld2    {v6.4s, v7.4s}, [%10]       \n"// v6

									"fmla   v8.4s, v5.4s, v0.s[0]       \n"
									"fmla   v9.4s, v5.4s, v0.s[1]       \n"
									"fmla   v10.4s, v5.4s, v0.s[2]      \n"
									"fmla   v11.4s, v5.4s, v0.s[3]      \n"

									"ld1    {v2.4s, v3.4s}, [%12], #32  \n"

									"ext    v6.16b, v4.16b, v6.16b, #4  \n"// v6=12

									"fmla   v12.4s, v5.4s, v1.s[0]      \n"
									"fmla   v13.4s, v5.4s, v1.s[1]      \n"
									"fmla   v14.4s, v5.4s, v1.s[2]      \n"
									"fmla   v15.4s, v5.4s, v1.s[3]      \n"

									///
									"prfm   pldl1keep, [%11, #256]      \n"
									"ld2    {v4.4s, v5.4s}, [%11], #32  \n"// v4=20 v5=21

									"fmla   v8.4s, v6.4s, v2.s[0]       \n"
									"fmla   v9.4s, v6.4s, v2.s[1]       \n"
									"fmla   v10.4s, v6.4s, v2.s[2]      \n"
									"fmla   v11.4s, v6.4s, v2.s[3]      \n"

									"ld1    {v0.4s, v1.4s}, [%12], #32  \n"

									"fmla   v12.4s, v6.4s, v3.s[0]      \n"
									"fmla   v13.4s, v6.4s, v3.s[1]      \n"
									"fmla   v14.4s, v6.4s, v3.s[2]      \n"
									"fmla   v15.4s, v6.4s, v3.s[3]      \n"

									"fmla   v8.4s, v4.4s, v0.s[0]       \n"
									"fmla   v9.4s, v4.4s, v0.s[1]       \n"
									"fmla   v10.4s, v4.4s, v0.s[2]      \n"
									"fmla   v11.4s, v4.4s, v0.s[3]      \n"

									"ld1    {v2.4s, v3.4s}, [%12], #32  \n"

									"fmla   v12.4s, v4.4s, v1.s[0]      \n"
									"fmla   v13.4s, v4.4s, v1.s[1]      \n"
									"fmla   v14.4s, v4.4s, v1.s[2]      \n"
									"fmla   v15.4s, v4.4s, v1.s[3]      \n"

									"prfm   pldl1keep, [%11, #256]      \n"
									"ld2    {v6.4s, v7.4s}, [%11]       \n"// v6

									"fmla   v8.4s, v5.4s, v2.s[0]       \n"
									"fmla   v9.4s, v5.4s, v2.s[1]       \n"
									"fmla   v10.4s, v5.4s, v2.s[2]      \n"
									"fmla   v11.4s, v5.4s, v2.s[3]      \n"

									"ext    v6.16b, v4.16b, v6.16b, #4  \n"// v6=22

									"ld1    {v0.4s, v1.4s}, [%12], #32  \n"

									"fmla   v12.4s, v5.4s, v3.s[0]      \n"
									"fmla   v13.4s, v5.4s, v3.s[1]      \n"
									"fmla   v14.4s, v5.4s, v3.s[2]      \n"
									"fmla   v15.4s, v5.4s, v3.s[3]      \n"

									"fmla   v8.4s, v6.4s, v0.s[0]       \n"
									"fmla   v9.4s, v6.4s, v0.s[1]       \n"
									"fmla   v10.4s, v6.4s, v0.s[2]      \n"
									"fmla   v11.4s, v6.4s, v0.s[3]      \n"

									"fmla   v12.4s, v6.4s, v1.s[0]      \n"
									"fmla   v13.4s, v6.4s, v1.s[1]      \n"

									"st1    {v8.4s}, [%1], #16          \n"
									"st1    {v9.4s}, [%2], #16          \n"

									"fmla   v14.4s, v6.4s, v1.s[2]      \n"
									"fmla   v15.4s, v6.4s, v1.s[3]      \n"

									"st1    {v10.4s}, [%3], #16         \n"
									"st1    {v11.4s}, [%4], #16         \n"

									"sub    %12, %12, #288              \n"

									"st1    {v12.4s}, [%5], #16         \n"
									"st1    {v13.4s}, [%6], #16         \n"

									"subs   %w0, %w0, #1                \n"

									"st1    {v14.4s}, [%7], #16         \n"
									"st1    {v15.4s}, [%8], #16         \n"

									"bne    0b                          \n"
									: "=r"(nn),         // %0
									"=r"(outptr0),    // %1
									"=r"(outptr1),    // %2
									"=r"(outptr2),    // %3
									"=r"(outptr3),    // %4
									"=r"(outptr4),    // %5
									"=r"(outptr5),    // %6
									"=r"(outptr6),    // %7
									"=r"(outptr7),    // %8
									"=r"(r0),         // %9
									"=r"(r1),         // %10
									"=r"(r2),         // %11
									"=r"(ktmp)        // %12
									: "0"(nn),
									"1"(outptr0),
									"2"(outptr1),
									"3"(outptr2),
									"4"(outptr3),
									"5"(outptr4),
									"6"(outptr5),
									"7"(outptr6),
									"8"(outptr7),
									"9"(r0),
									"10"(r1),
									"11"(r2),
									"12"(ktmp)
									: "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
									);
							}
#else // __aarch64__
							if (nn > 0)
							{
								asm volatile(
									"0:                             \n"

									"pld        [%1, #128]          \n"
									"vld1.f32   {d16-d17}, [%1]     \n"
									"pld        [%2, #128]          \n"
									"vld1.f32   {d18-d19}, [%2]     \n"

									"pld        [%3, #128]          \n"
									"vld1.f32   {d20-d21}, [%3]     \n"
									"pld        [%4, #128]          \n"
									"vld1.f32   {d22-d23}, [%4]     \n"

									///
									"pld        [%9, #256]          \n"
									"vld2.f32   {d8-d11}, [%9]!     \n"// q4=00 q5=01

									"vld1.f32   {d0-d3}, [%12 :128]! \n"

									"vmla.f32   q8, q4, d0[0]       \n"
									"vmla.f32   q9, q4, d0[1]       \n"

									"pld        [%5, #128]          \n"
									"vld1.f32   {d24-d25}, [%5]     \n"
									"pld        [%6, #128]          \n"
									"vld1.f32   {d26-d27}, [%6]     \n"

									"vmla.f32   q10, q4, d1[0]      \n"
									"vmla.f32   q11, q4, d1[1]      \n"

									"pld        [%7, #128]          \n"
									"vld1.f32   {d28-d29}, [%7]     \n"
									"pld        [%8, #128]          \n"
									"vld1.f32   {d30-d31}, [%8]     \n"

									"vld1.f32   {d4-d7}, [%12 :128]! \n"

									"vmla.f32   q12, q4, d2[0]      \n"
									"vmla.f32   q13, q4, d2[1]      \n"
									"vmla.f32   q14, q4, d3[0]      \n"
									"vmla.f32   q15, q4, d3[1]      \n"

									"pld        [%9, #128]          \n"
									"vld2.f32   {d12-d13}, [%9]     \n"// q6

									"vmla.f32   q8, q5, d4[0]       \n"
									"vmla.f32   q9, q5, d4[1]       \n"
									"vmla.f32   q10, q5, d5[0]      \n"
									"vmla.f32   q11, q5, d5[1]      \n"

									"vext.f32   q6, q4, q6, #1      \n"// q6=02

									"vld1.f32   {d0-d3}, [%12 :128]! \n"

									"vmla.f32   q12, q5, d6[0]      \n"
									"vmla.f32   q13, q5, d6[1]      \n"
									"vmla.f32   q14, q5, d7[0]      \n"
									"vmla.f32   q15, q5, d7[1]      \n"

									///
									"pld        [%10, #256]         \n"
									"vld2.f32   {d8-d11}, [%10]!    \n"// q4=10 q5=11

									"vmla.f32   q8, q6, d0[0]       \n"
									"vmla.f32   q9, q6, d0[1]       \n"
									"vmla.f32   q10, q6, d1[0]      \n"
									"vmla.f32   q11, q6, d1[1]      \n"

									"vld1.f32   {d4-d7}, [%12 :128]! \n"

									"vmla.f32   q12, q6, d2[0]      \n"
									"vmla.f32   q13, q6, d2[1]      \n"
									"vmla.f32   q14, q6, d3[0]      \n"
									"vmla.f32   q15, q6, d3[1]      \n"

									"vmla.f32   q8, q4, d4[0]       \n"
									"vmla.f32   q9, q4, d4[1]       \n"
									"vmla.f32   q10, q4, d5[0]      \n"
									"vmla.f32   q11, q4, d5[1]      \n"

									"vld1.f32   {d0-d3}, [%12 :128]! \n"

									"vmla.f32   q12, q4, d6[0]      \n"
									"vmla.f32   q13, q4, d6[1]      \n"
									"vmla.f32   q14, q4, d7[0]      \n"
									"vmla.f32   q15, q4, d7[1]      \n"

									"pld        [%10, #128]         \n"
									"vld2.f32   {d12-d13}, [%10]    \n"// q6

									"vmla.f32   q8, q5, d0[0]       \n"
									"vmla.f32   q9, q5, d0[1]       \n"
									"vmla.f32   q10, q5, d1[0]      \n"
									"vmla.f32   q11, q5, d1[1]      \n"

									"vld1.f32   {d4-d7}, [%12 :128]! \n"

									"vext.f32   q6, q4, q6, #1      \n"// q6=12

									"vmla.f32   q12, q5, d2[0]      \n"
									"vmla.f32   q13, q5, d2[1]      \n"
									"vmla.f32   q14, q5, d3[0]      \n"
									"vmla.f32   q15, q5, d3[1]      \n"

									///
									"pld        [%11, #256]         \n"
									"vld2.f32   {d8-d11}, [%11]!    \n"// q4=20 q5=21

									"vmla.f32   q8, q6, d4[0]       \n"
									"vmla.f32   q9, q6, d4[1]       \n"
									"vmla.f32   q10, q6, d5[0]      \n"
									"vmla.f32   q11, q6, d5[1]      \n"

									"vld1.f32   {d0-d3}, [%12 :128]! \n"

									"vmla.f32   q12, q6, d6[0]      \n"
									"vmla.f32   q13, q6, d6[1]      \n"
									"vmla.f32   q14, q6, d7[0]      \n"
									"vmla.f32   q15, q6, d7[1]      \n"

									"vmla.f32   q8, q4, d0[0]       \n"
									"vmla.f32   q9, q4, d0[1]       \n"
									"vmla.f32   q10, q4, d1[0]      \n"
									"vmla.f32   q11, q4, d1[1]      \n"

									"vld1.f32   {d4-d7}, [%12 :128]! \n"

									"vmla.f32   q12, q4, d2[0]      \n"
									"vmla.f32   q13, q4, d2[1]      \n"
									"vmla.f32   q14, q4, d3[0]      \n"
									"vmla.f32   q15, q4, d3[1]      \n"

									"pld        [%11, #128]         \n"
									"vld2.f32   {d12-d13}, [%11]    \n"// q6

									"vmla.f32   q8, q5, d4[0]       \n"
									"vmla.f32   q9, q5, d4[1]       \n"
									"vmla.f32   q10, q5, d5[0]      \n"
									"vmla.f32   q11, q5, d5[1]      \n"

									"vext.f32   q6, q4, q6, #1      \n"// q6=22

									"vld1.f32   {d0-d3}, [%12 :128]! \n"

									"vmla.f32   q12, q5, d6[0]      \n"
									"vmla.f32   q13, q5, d6[1]      \n"
									"vmla.f32   q14, q5, d7[0]      \n"
									"vmla.f32   q15, q5, d7[1]      \n"

									"vmla.f32   q8, q6, d0[0]       \n"
									"vmla.f32   q9, q6, d0[1]       \n"
									"vmla.f32   q10, q6, d1[0]      \n"
									"vmla.f32   q11, q6, d1[1]      \n"

									"vmla.f32   q12, q6, d2[0]      \n"
									"vmla.f32   q13, q6, d2[1]      \n"

									"vst1.f32   {d16-d17}, [%1]!    \n"
									"vst1.f32   {d18-d19}, [%2]!    \n"

									"vmla.f32   q14, q6, d3[0]      \n"
									"vmla.f32   q15, q6, d3[1]      \n"

									"vst1.f32   {d20-d21}, [%3]!    \n"
									"vst1.f32   {d22-d23}, [%4]!    \n"

									"sub        %12, %12, #288      \n"

									"vst1.f32   {d24-d25}, [%5]!    \n"
									"vst1.f32   {d26-d27}, [%6]!    \n"

									"subs       %0, #1              \n"

									"vst1.f32   {d28-d29}, [%7]!    \n"
									"vst1.f32   {d30-d31}, [%8]!    \n"

									"bne        0b                  \n"
									: "=r"(nn),         // %0
									"=r"(outptr0),    // %1
									"=r"(outptr1),    // %2
									"=r"(outptr2),    // %3
									"=r"(outptr3),    // %4
									"=r"(outptr4),    // %5
									"=r"(outptr5),    // %6
									"=r"(outptr6),    // %7
									"=r"(outptr7),    // %8
									"=r"(r0),         // %9
									"=r"(r1),         // %10
									"=r"(r2),         // %11
									"=r"(ktmp)        // %12
									: "0"(nn),
									"1"(outptr0),
									"2"(outptr1),
									"3"(outptr2),
									"4"(outptr3),
									"5"(outptr4),
									"6"(outptr5),
									"7"(outptr6),
									"8"(outptr7),
									"9"(r0),
									"10"(r1),
									"11"(r2),
									"12"(ktmp)
									: "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
									);
							}
#endif // __aarch64__
#endif // __ARM_NEON
							for (; remain > 0; remain--)
							{
#if __ARM_NEON
#if __aarch64__
								asm volatile(
									"ld1    {v10.4s, v11.4s}, [%11], #32    \n"

									"prfm   pldl1keep, [%8, #128]   \n"
									"ld1    {v0.4s}, [%8]           \n"

									"ld1    {v12.4s, v13.4s}, [%11], #32    \n"

									"ld1    {v8.s}[0], [%0]         \n"
									"ld1    {v8.s}[1], [%1]         \n"
									"ld1    {v8.s}[2], [%2]         \n"
									"ld1    {v8.s}[3], [%3]         \n"

									"fmul   v14.4s, v10.4s, v0.s[0] \n"
									"fmul   v15.4s, v11.4s, v0.s[0] \n"

									"ld1    {v9.s}[0], [%4]         \n"
									"ld1    {v9.s}[1], [%5]         \n"
									"ld1    {v9.s}[2], [%6]         \n"
									"ld1    {v9.s}[3], [%7]         \n"

									"ld1    {v10.4s, v11.4s}, [%11], #32    \n"

									"fmla   v8.4s, v12.4s, v0.s[1]  \n"
									"fmla   v9.4s, v13.4s, v0.s[1]  \n"

									"ld1    {v12.4s, v13.4s}, [%11], #32    \n"

									"fmla   v14.4s, v10.4s, v0.s[2] \n"
									"fmla   v15.4s, v11.4s, v0.s[2] \n"

									"prfm   pldl1keep, [%9, #128]   \n"
									"ld1    {v1.4s}, [%9]           \n"

									"ld1    {v10.4s, v11.4s}, [%11], #32    \n"

									"fmla   v8.4s, v12.4s, v1.s[0]  \n"
									"fmla   v9.4s, v13.4s, v1.s[0]  \n"

									"ld1    {v12.4s, v13.4s}, [%11], #32    \n"

									"fmla   v14.4s, v10.4s, v1.s[1] \n"
									"fmla   v15.4s, v11.4s, v1.s[1] \n"

									"ld1    {v10.4s, v11.4s}, [%11], #32    \n"

									"fmla   v8.4s, v12.4s, v1.s[2]  \n"
									"fmla   v9.4s, v13.4s, v1.s[2]  \n"

									"prfm   pldl1keep, [%10, #128]  \n"
									"ld1    {v0.4s}, [%10]          \n"

									"ld1    {v12.4s, v13.4s}, [%11], #32    \n"

									"fmla   v14.4s, v10.4s, v0.s[0] \n"
									"fmla   v15.4s, v11.4s, v0.s[0] \n"

									"ld1    {v10.4s, v11.4s}, [%11], #32    \n"

									"fmla   v8.4s, v12.4s, v0.s[1]  \n"
									"fmla   v9.4s, v13.4s, v0.s[1]  \n"

									"fmla   v14.4s, v10.4s, v0.s[2] \n"
									"fmla   v15.4s, v11.4s, v0.s[2] \n"

									"fadd   v8.4s, v8.4s, v14.4s    \n"
									"fadd   v9.4s, v9.4s, v15.4s    \n"

									"sub    %11, %11, #288          \n"

									"st1    {v8.s}[0], [%0], #4     \n"
									"st1    {v8.s}[1], [%1], #4     \n"
									"st1    {v8.s}[2], [%2], #4     \n"
									"st1    {v8.s}[3], [%3], #4     \n"

									"st1    {v9.s}[0], [%4], #4     \n"
									"st1    {v9.s}[1], [%5], #4     \n"
									"st1    {v9.s}[2], [%6], #4     \n"
									"st1    {v9.s}[3], [%7], #4     \n"

									: "=r"(outptr0),    // %0
									"=r"(outptr1),    // %1
									"=r"(outptr2),    // %2
									"=r"(outptr3),    // %3
									"=r"(outptr4),    // %4
									"=r"(outptr5),    // %5
									"=r"(outptr6),    // %6
									"=r"(outptr7),    // %7
									"=r"(r0),         // %8
									"=r"(r1),         // %9
									"=r"(r2),         // %10
									"=r"(ktmp)        // %11
									: "0"(outptr0),
									"1"(outptr1),
									"2"(outptr2),
									"3"(outptr3),
									"4"(outptr4),
									"5"(outptr5),
									"6"(outptr6),
									"7"(outptr7),
									"8"(r0),
									"9"(r1),
									"10"(r2),
									"11"(ktmp)
									: "memory", "v0", "v1", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
									);
#else // __aarch64__
								asm volatile(
									"vld1.f32   {d20-d23}, [%11 :128]! \n"

									"pld        [%8, #128]      \n"
									"vld1.f32   {d0-d1}, [%8]   \n"

									"vld1.f32   {d24-d27}, [%11 :128]! \n"

									"vld1.f32   {d16[0]}, [%0]  \n"
									"vld1.f32   {d16[1]}, [%1]  \n"
									"vld1.f32   {d17[0]}, [%2]  \n"
									"vld1.f32   {d17[1]}, [%3]  \n"

									"vmul.f32   q14, q10, d0[0] \n"
									"vmul.f32   q15, q11, d0[0] \n"

									"vld1.f32   {d18[0]}, [%4]  \n"
									"vld1.f32   {d18[1]}, [%5]  \n"
									"vld1.f32   {d19[0]}, [%6]  \n"
									"vld1.f32   {d19[1]}, [%7]  \n"

									"vld1.f32   {d20-d23}, [%11 :128]! \n"

									"vmla.f32   q8, q12, d0[1]  \n"
									"vmla.f32   q9, q13, d0[1]  \n"

									"vld1.f32   {d24-d27}, [%11 :128]! \n"

									"vmla.f32   q14, q10, d1[0] \n"
									"vmla.f32   q15, q11, d1[0] \n"

									"pld        [%9, #128]      \n"
									"vld1.f32   {d2-d3}, [%9]   \n"

									"vld1.f32   {d20-d23}, [%11 :128]! \n"

									"vmla.f32   q8, q12, d2[0]  \n"
									"vmla.f32   q9, q13, d2[0]  \n"

									"vld1.f32   {d24-d27}, [%11 :128]! \n"

									"vmla.f32   q14, q10, d2[1] \n"
									"vmla.f32   q15, q11, d2[1] \n"

									"vld1.f32   {d20-d23}, [%11 :128]! \n"

									"vmla.f32   q8, q12, d3[0]  \n"
									"vmla.f32   q9, q13, d3[0]  \n"

									"pld        [%10, #128]     \n"
									"vld1.f32   {d0-d1}, [%10]  \n"

									"vld1.f32   {d24-d27}, [%11 :128]! \n"

									"vmla.f32   q14, q10, d0[0] \n"
									"vmla.f32   q15, q11, d0[0] \n"

									"vld1.f32   {d20-d23}, [%11 :128]! \n"

									"vmla.f32   q8, q12, d0[1]  \n"
									"vmla.f32   q9, q13, d0[1]  \n"

									"vmla.f32   q14, q10, d1[0] \n"
									"vmla.f32   q15, q11, d1[0] \n"

									"vadd.f32   q8, q8, q14     \n"
									"vadd.f32   q9, q9, q15     \n"

									"sub        %11, %11, #288  \n"

									"vst1.f32   {d16[0]}, [%0]! \n"
									"vst1.f32   {d16[1]}, [%1]! \n"
									"vst1.f32   {d17[0]}, [%2]! \n"
									"vst1.f32   {d17[1]}, [%3]! \n"

									"vst1.f32   {d18[0]}, [%4]! \n"
									"vst1.f32   {d18[1]}, [%5]! \n"
									"vst1.f32   {d19[0]}, [%6]! \n"
									"vst1.f32   {d19[1]}, [%7]! \n"

									: "=r"(outptr0),    // %0
									"=r"(outptr1),    // %1
									"=r"(outptr2),    // %2
									"=r"(outptr3),    // %3
									"=r"(outptr4),    // %4
									"=r"(outptr5),    // %5
									"=r"(outptr6),    // %6
									"=r"(outptr7),    // %7
									"=r"(r0),         // %8
									"=r"(r1),         // %9
									"=r"(r2),         // %10
									"=r"(ktmp)        // %11
									: "0"(outptr0),
									"1"(outptr1),
									"2"(outptr2),
									"3"(outptr3),
									"4"(outptr4),
									"5"(outptr5),
									"6"(outptr6),
									"7"(outptr7),
									"8"(r0),
									"9"(r1),
									"10"(r2),
									"11"(ktmp)
									: "memory", "q0", "q1", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
									);
#endif // __aarch64__
#else // __ARM_NEON
								float sum0 = 0.f;
								float sum1 = 0.f;
								float sum2 = 0.f;
								float sum3 = 0.f;
								float sum4 = 0.f;
								float sum5 = 0.f;
								float sum6 = 0.f;
								float sum7 = 0.f;

								sum0 += r0[0] * ktmp[0];
								sum1 += r0[0] * ktmp[1];
								sum2 += r0[0] * ktmp[2];
								sum3 += r0[0] * ktmp[3];
								sum4 += r0[0] * ktmp[4];
								sum5 += r0[0] * ktmp[5];
								sum6 += r0[0] * ktmp[6];
								sum7 += r0[0] * ktmp[7];
								ktmp += 8;

								sum0 += r0[1] * ktmp[0];
								sum1 += r0[1] * ktmp[1];
								sum2 += r0[1] * ktmp[2];
								sum3 += r0[1] * ktmp[3];
								sum4 += r0[1] * ktmp[4];
								sum5 += r0[1] * ktmp[5];
								sum6 += r0[1] * ktmp[6];
								sum7 += r0[1] * ktmp[7];
								ktmp += 8;

								sum0 += r0[2] * ktmp[0];
								sum1 += r0[2] * ktmp[1];
								sum2 += r0[2] * ktmp[2];
								sum3 += r0[2] * ktmp[3];
								sum4 += r0[2] * ktmp[4];
								sum5 += r0[2] * ktmp[5];
								sum6 += r0[2] * ktmp[6];
								sum7 += r0[2] * ktmp[7];
								ktmp += 8;

								sum0 += r1[0] * ktmp[0];
								sum1 += r1[0] * ktmp[1];
								sum2 += r1[0] * ktmp[2];
								sum3 += r1[0] * ktmp[3];
								sum4 += r1[0] * ktmp[4];
								sum5 += r1[0] * ktmp[5];
								sum6 += r1[0] * ktmp[6];
								sum7 += r1[0] * ktmp[7];
								ktmp += 8;

								sum0 += r1[1] * ktmp[0];
								sum1 += r1[1] * ktmp[1];
								sum2 += r1[1] * ktmp[2];
								sum3 += r1[1] * ktmp[3];
								sum4 += r1[1] * ktmp[4];
								sum5 += r1[1] * ktmp[5];
								sum6 += r1[1] * ktmp[6];
								sum7 += r1[1] * ktmp[7];
								ktmp += 8;

								sum0 += r1[2] * ktmp[0];
								sum1 += r1[2] * ktmp[1];
								sum2 += r1[2] * ktmp[2];
								sum3 += r1[2] * ktmp[3];
								sum4 += r1[2] * ktmp[4];
								sum5 += r1[2] * ktmp[5];
								sum6 += r1[2] * ktmp[6];
								sum7 += r1[2] * ktmp[7];
								ktmp += 8;

								sum0 += r2[0] * ktmp[0];
								sum1 += r2[0] * ktmp[1];
								sum2 += r2[0] * ktmp[2];
								sum3 += r2[0] * ktmp[3];
								sum4 += r2[0] * ktmp[4];
								sum5 += r2[0] * ktmp[5];
								sum6 += r2[0] * ktmp[6];
								sum7 += r2[0] * ktmp[7];
								ktmp += 8;

								sum0 += r2[1] * ktmp[0];
								sum1 += r2[1] * ktmp[1];
								sum2 += r2[1] * ktmp[2];
								sum3 += r2[1] * ktmp[3];
								sum4 += r2[1] * ktmp[4];
								sum5 += r2[1] * ktmp[5];
								sum6 += r2[1] * ktmp[6];
								sum7 += r2[1] * ktmp[7];
								ktmp += 8;

								sum0 += r2[2] * ktmp[0];
								sum1 += r2[2] * ktmp[1];
								sum2 += r2[2] * ktmp[2];
								sum3 += r2[2] * ktmp[3];
								sum4 += r2[2] * ktmp[4];
								sum5 += r2[2] * ktmp[5];
								sum6 += r2[2] * ktmp[6];
								sum7 += r2[2] * ktmp[7];
								ktmp += 8;

								*outptr0 += sum0;
								*outptr1 += sum1;
								*outptr2 += sum2;
								*outptr3 += sum3;
								*outptr4 += sum4;
								*outptr5 += sum5;
								*outptr6 += sum6;
								*outptr7 += sum7;

								ktmp -= 8 * 9;

								outptr0++;
								outptr1++;
								outptr2++;
								outptr3++;
								outptr4++;
								outptr5++;
								outptr6++;
								outptr7++;
#endif // __ARM_NEON
								r0 += 2;
								r1 += 2;
								r2 += 2;
							}

							r0 += tailstep;
							r1 += tailstep;
							r2 += tailstep;
						}

						ktmp += 8 * 9;
					}
				}

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
				for (int p = remain_outch_start; p < outch; p++)
				{
					float *out = top_data + (p)* top_cstep;

					const float bias0 = this->bias_term_ ? bias[p] : 0.f;

					fill(out, top_cstep, bias0);

					const float* ktmp = kernel_data + (p / 8 + p % 8) * kernel_cstep;

					for (int q = 0; q < inch; q++)
					{
						float* outptr = out;

						const float* img0 = bottom_data + (q)* bottom_cstep;

						const float* r0 = img0;
						const float* r1 = img0 + w;
						const float* r2 = img0 + w * 2;

						const float* k0 = ktmp;
						const float* k1 = ktmp + 3;
						const float* k2 = ktmp + 6;

#if __ARM_NEON
						float32x4_t _k0123 = vld1q_f32(k0);
						float32x4_t _k3456 = vld1q_f32(k1);
						float32x4_t _k6789 = vld1q_f32(k2);
#endif // __ARM_NEON

						int i = 0;

						for (; i < outh; i++)
						{
#if __ARM_NEON
							int nn = outw >> 2;
							int remain = outw & 3;
#else
							int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
							if (nn > 0)
							{
								asm volatile(
									"prfm       pldl1keep, [%2, #256]          \n"
									"ld2        {v2.4s, v3.4s}, [%2], #32      \n"
									"0:                                        \n"

									"prfm       pldl1keep, [%1, #128]          \n"
									"ld1        {v0.4s}, [%1]                  \n"

									"fmla       v0.4s,  v2.4s, %10.s[0]        \n"
									"fmul       v10.4s, v3.4s, %10.s[1]        \n"

									"prfm       pldl1keep, [%2, #256]          \n"
									"ld2        {v8.4s, v9.4s}, [%2]           \n"
									"ext        v1.16b, v2.16b, v8.16b, #4     \n"

									"fmul       v11.4s, v1.4s, %10.s[2]        \n"

									"prfm       pldl1keep, [%3, #256]          \n"
									"ld2        {v2.4s, v3.4s}, [%3], #32      \n"

									"fmla       v0.4s,  v2.4s, %11.s[0]        \n"
									"fmla       v10.4s, v3.4s, %11.s[1]        \n"

									"prfm       pldl1keep, [%3, #256]          \n"
									"ld2        {v8.4s, v9.4s}, [%3]           \n"
									"ext        v1.16b, v2.16b, v8.16b, #4     \n"

									"fmla       v11.4s, v1.4s, %11.s[2]        \n"

									"prfm       pldl1keep, [%4, #256]          \n"
									"ld2        {v2.4s, v3.4s}, [%4], #32      \n"

									"fmla       v0.4s,  v2.4s, %12.s[0]        \n"
									"fmla       v10.4s, v3.4s, %12.s[1]        \n"

									"prfm       pldl1keep, [%4, #256]          \n"
									"ld2        {v8.4s, v9.4s}, [%4]           \n"
									"ext        v1.16b, v2.16b, v8.16b, #4     \n"

									"fmla       v11.4s, v1.4s, %12.s[2]        \n"

									"prfm       pldl1keep, [%2, #256]          \n"
									"ld2        {v2.4s, v3.4s}, [%2], #32      \n"

									"fadd       v0.4s, v0.4s, v10.4s           \n"
									"fadd       v0.4s, v0.4s, v11.4s           \n"

									"subs       %w0, %w0, #1                   \n"
									"st1        {v0.4s}, [%1], #16             \n"
									"bne        0b                             \n"
									"sub        %2, %2, #32                    \n"
									: "=r"(nn),     // %0
									"=r"(outptr), // %1
									"=r"(r0),     // %2
									"=r"(r1),     // %3
									"=r"(r2)      // %4
									: "0"(nn),
									"1"(outptr),
									"2"(r0),
									"3"(r1),
									"4"(r2),
									"w"(_k0123),  // %10
									"w"(_k3456),  // %11
									"w"(_k6789)   // %12
									: "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
									);
							}
#else
							if (nn > 0)
							{
								asm volatile(
									"pld        [%2, #256]          \n"
									"vld2.f32   {d4-d7}, [%2]!      \n"

									"0:                             \n"
									"pld        [%1, #128]          \n"
									"vld1.f32   {d0-d1}, [%1]       \n"

									"vmla.f32   q0, q2, %e10[0]     \n"
									"vmul.f32   q10, q3, %e10[1]    \n"

									"pld        [%2, #128]          \n"
									"vld2.f32   {d16-d17}, [%2]     \n"
									"vext.32    q1, q2, q8, #1      \n"

									"vmul.f32   q11, q1, %f10[0]    \n"

									"pld        [%3, #256]          \n"
									"vld2.f32   {d4-d7}, [%3]!      \n"

									"vmla.f32   q0, q2, %e11[0]     \n"
									"vmla.f32   q10, q3, %e11[1]    \n"

									"pld        [%3, #128]          \n"
									"vld2.f32   {d16-d17}, [%3]     \n"
									"vext.32    q1, q2, q8, #1      \n"

									"vmla.f32   q11, q1, %f11[0]    \n"

									"pld        [%4, #256]          \n"
									"vld2.f32   {d4-d7}, [%4]!      \n"

									"vmla.f32   q0, q2, %e12[0]     \n"
									"vmla.f32   q10, q3, %e12[1]    \n"

									"pld        [%4, #128]          \n"
									"vld2.f32   {d16-d17}, [%4]     \n"
									"vext.32    q1, q2, q8, #1      \n"

									"vmla.f32   q11, q1, %f12[0]    \n"

									"pld        [%2, #256]          \n"
									"vld2.f32   {d4-d7}, [%2]!      \n"

									"vadd.f32   q0, q0, q10         \n"
									"vadd.f32   q0, q0, q11         \n"

									"subs       %0, #1              \n"
									"vst1.f32   {d0-d1}, [%1]!      \n"
									"bne        0b                  \n"
									"sub        %2, #32             \n"
									: "=r"(nn),     // %0
									"=r"(outptr), // %1
									"=r"(r0),     // %2
									"=r"(r1),     // %3
									"=r"(r2)      // %4
									: "0"(nn),
									"1"(outptr),
									"2"(r0),
									"3"(r1),
									"4"(r2),
									"w"(_k0123),  // %10
									"w"(_k3456),  // %11
									"w"(_k6789)   // %12
									: "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
									);
							}
#endif // __aarch64__
#endif // __ARM_NEON
							for (; remain > 0; remain--)
							{
#if __ARM_NEON
								float32x4_t _r00 = vld1q_f32(r0);
								float32x4_t _r10 = vld1q_f32(r1);
								float32x4_t _r20 = vld1q_f32(r2);

								float32x4_t _sum = vmulq_f32(_r00, _k0123);
								_sum = vmlaq_f32(_sum, _r10, _k3456);
								_sum = vmlaq_f32(_sum, _r20, _k6789);

								_sum = vsetq_lane_f32(*outptr, _sum, 3);

#if __aarch64__
								*outptr = vaddvq_f32(_sum);
#else
								float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
								_ss = vpadd_f32(_ss, _ss);

								*outptr = vget_lane_f32(_ss, 0);
#endif // __aarch64__
#else
								float sum = 0;

								sum += r0[0] * ktmp[0];
								sum += r0[1] * ktmp[1];
								sum += r0[2] * ktmp[2];
								sum += r1[0] * ktmp[3];
								sum += r1[1] * ktmp[4];
								sum += r1[2] * ktmp[5];
								sum += r2[0] * ktmp[6];
								sum += r2[1] * ktmp[7];
								sum += r2[2] * ktmp[8];

								*outptr += sum;
#endif // __ARM_NEON

								r0 += 2;
								r1 += 2;
								r2 += 2;
								outptr++;
							}

							r0 += tailstep;
							r1 += tailstep;
							r2 += tailstep;
						}

						ktmp += 9;
					}
				}
			}
		}

		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv_im2col_sgemm_neon(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top)
		{
			int num = bottom->num();
			int w = bottom->width();
			int h = bottom->height();
			int inch = bottom->channels();
			int bottom_cstep = w * h;

			int outw = top->width();
			int outh = top->height();
			int outch = top->channels();

			const float *kernel_tm_gemm_data = kernel_tm_gemm_->cpu_data();
			int kernel_tm_gemm_cstep = kernel_tm_gemm_->width() * kernel_tm_gemm_->height();

			const float* bias = nullptr;
			if (this->bias_term_)
				bias = this->weights_f32_[1]->cpu_data();

			int top_cstep = outw * outh;

			int kernel_size = this->kernel_size_h_ * this->kernel_size_w_;
			int out_size = outw * outh;
			// im2col
			memory::tensor<float> bottom_im2col(std::vector<int>{1, 1, kernel_size*inch, out_size}, -1, memory::NCHW);
			float* ret = bottom_im2col.mutable_cpu_data();
			
			const float *bottom_im2col_data = bottom_im2col.cpu_data();

			// bottom_im2col memory packed 8 x 8
			memory::tensor<float> bottom_tm(std::vector<int>{1, out_size / 8 + out_size % 8, inch, 8 * kernel_size}, -1, memory::NCHW);
			float *bottom_tm_data = bottom_tm.mutable_cpu_data();
			int bottom_tm_cstep = bottom_tm.width() * bottom_tm.height();

			for (int num_i = 0; num_i < num; num_i++)
			{
				const float *bottom_data = bottom->cpu_data() + num_i * inch * bottom_cstep;
				float* top_data = top->mutable_cpu_data() + num_i * outch * top_cstep;

				{
					const int stride = kernel_size*out_size;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
					for (int p = 0; p < inch; p++)
					{
						const float* input = bottom_data + (p)* bottom_cstep;
						int retID = stride * p;
						for (int u = 0; u < this->kernel_size_h_; u++)
						{
							for (int v = 0; v < this->kernel_size_w_; v++)
							{
								for (int i = 0; i < outh; i++)
								{
									for (int j = 0; j < outw; j++)
									{
										int row = u + i * this->stride_h_;
										int col = v + j * this->stride_w_;
										int index = row * w + col;
										ret[retID] = input[index];
										retID++;
									}
								}
							}
						}
					}
				}


				{
					int nn_size = out_size >> 3;
					int remain_size_start = nn_size << 3;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
					for (int ii = 0; ii < nn_size; ii++)
					{
						int i = ii * 8;

						const float* img0 = bottom_im2col_data;
						img0 += i;

						float* tmpptr = bottom_tm_data + (i / 8) * bottom_tm_cstep;

						for (int q = 0; q < inch*kernel_size; q++)
						{
#if __ARM_NEON
#if __aarch64__
							asm volatile(
								"prfm    pldl1keep, [%0, #256]   \n"
								"ld1     {v0.4s, v1.4s}, [%0]    \n"
								"st1     {v0.4s, v1.4s}, [%1]    \n"
								: "=r"(img0),   // %0
								"=r"(tmpptr)  // %1
								: "0"(img0),
								"1"(tmpptr)
								: "cc", "memory", "v0", "v1"
								);
#else
							asm volatile(
								"pld        [%0, #256]          \n"
								"vld1.f32   {d0-d3}, [%0]       \n"
								"vst1.f32   {d0-d3}, [%1]       \n"
								: "=r"(img0),   // %0
								"=r"(tmpptr)  // %1
								: "0"(img0),
								"1"(tmpptr)
								: "memory", "q0", "q1"
								);
#endif // __aarch64__
#else                
							tmpptr[0] = img0[0];
							tmpptr[1] = img0[1];
							tmpptr[2] = img0[2];
							tmpptr[3] = img0[3];
							tmpptr[4] = img0[4];
							tmpptr[5] = img0[5];
							tmpptr[6] = img0[6];
							tmpptr[7] = img0[7];
#endif // __ARM_NEON              
							tmpptr += 8;
							img0 += out_size;
						}
					}

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
					for (int i = remain_size_start; i < out_size; i++)
					{
						const float* img0 = bottom_im2col_data;
						img0 += i;

						float* tmpptr = bottom_tm_data + (i / 8 + i % 8) * bottom_tm_cstep;

						for (int q = 0; q < inch*kernel_size; q++)
						{
							tmpptr[0] = img0[0];

							tmpptr += 1;
							img0 += out_size;
						}
					}
				}

				// sgemm(int M, int N, int L, float* A, float* B, float* C)
				{
					//int M = outch;                    // outch
					int N = out_size;                // outsize or out stride
					int L = kernel_size * inch; // ksize * inch

					int nn_outch = 0;
					int remain_outch_start = 0;

#if __aarch64__
					nn_outch = outch >> 3;
					remain_outch_start = nn_outch << 3;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
					for (int pp = 0; pp < nn_outch; pp++)
					{
						int i = pp * 8;

						float* output0 = top_data + (i)* top_cstep;
						float* output1 = top_data + (i + 1) * top_cstep;
						float* output2 = top_data + (i + 2) * top_cstep;
						float* output3 = top_data + (i + 3) * top_cstep;
						float* output4 = top_data + (i + 4) * top_cstep;
						float* output5 = top_data + (i + 5) * top_cstep;
						float* output6 = top_data + (i + 6) * top_cstep;
						float* output7 = top_data + (i + 7) * top_cstep;

						const float zeros[8] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
						const float* biasptr = this->bias_term_ ? bias + i : zeros;

						int j = 0;
						for (; j + 7 < N; j = j + 8)
						{
							const float* vb = bottom_tm_data + (j / 8) * bottom_tm_cstep;
							const float* va = kernel_tm_gemm_data + (i / 8) * kernel_tm_gemm_cstep;
#if __ARM_NEON
							asm volatile(
								"ld1    {v0.4s, v1.4s}, [%21]   \n"
								"dup    v16.4s, v0.s[0]         \n"// sum0
								"dup    v17.4s, v0.s[0]         \n"
								"dup    v18.4s, v0.s[1]         \n"// sum1
								"dup    v19.4s, v0.s[1]         \n"
								"dup    v20.4s, v0.s[2]         \n"// sum2
								"dup    v21.4s, v0.s[2]         \n"
								"dup    v22.4s, v0.s[3]         \n"// sum3
								"dup    v23.4s, v0.s[3]         \n"
								"dup    v24.4s, v1.s[0]         \n"// sum4
								"dup    v25.4s, v1.s[0]         \n"
								"dup    v26.4s, v1.s[1]         \n"// sum5
								"dup    v27.4s, v1.s[1]         \n"
								"dup    v28.4s, v1.s[2]         \n"// sum6
								"dup    v29.4s, v1.s[2]         \n"
								"dup    v30.4s, v1.s[3]         \n"// sum7
								"dup    v31.4s, v1.s[3]         \n"

								"lsr         w4, %w20, #2            \n"// r4 = nn = L >> 2
								"cmp         w4, #0                  \n"
								"beq         1f                      \n"

								"0:                                  \n"// for (; k+3<L; k=k+4)

								"prfm   pldl1keep, [%9, #512]                       \n"
								"ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64     \n" // kernel
								"ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%9], #64     \n"

								"prfm   pldl1keep, [%8, #512]                       \n"
								"ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%8], #64   \n" // data
								"ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%8], #64 \n"
								// k0
								"fmla    v16.4s, v8.4s, v0.s[0]      \n"// sum0 += (a00-a70) * k00
								"fmla    v17.4s, v9.4s, v0.s[0]      \n"//
								"fmla    v18.4s, v8.4s, v0.s[1]      \n"// sum1 += (a00-a70) * k10
								"fmla    v19.4s, v9.4s, v0.s[1]      \n"//
								"fmla    v20.4s, v8.4s, v0.s[2]      \n"// sum2 += (a00-a70) * k20
								"fmla    v21.4s, v9.4s, v0.s[2]      \n"//
								"fmla    v22.4s, v8.4s, v0.s[3]      \n"// sum3 += (a00-a70) * k30
								"fmla    v23.4s, v9.4s, v0.s[3]      \n"//
								"fmla    v24.4s, v8.4s, v1.s[0]      \n"// sum4 += (a00-a70) * k40
								"fmla    v25.4s, v9.4s, v1.s[0]      \n"//
								"fmla    v26.4s, v8.4s, v1.s[1]      \n"// sum5 += (a00-a70) * k50
								"fmla    v27.4s, v9.4s, v1.s[1]      \n"//
								"fmla    v28.4s, v8.4s, v1.s[2]      \n"// sum6 += (a00-a70) * k60
								"fmla    v29.4s, v9.4s, v1.s[2]      \n"//
								"fmla    v30.4s, v8.4s, v1.s[3]      \n"// sum7 += (a00-a70) * k70
								"fmla    v31.4s, v9.4s, v1.s[3]      \n"//
																		// k1
								"fmla    v16.4s, v10.4s, v2.s[0]     \n"// sum0 += (a01-a71) * k01
								"fmla    v17.4s, v11.4s, v2.s[0]     \n"//
								"fmla    v18.4s, v10.4s, v2.s[1]     \n"// sum1 += (a01-a71) * k11
								"fmla    v19.4s, v11.4s, v2.s[1]     \n"//
								"fmla    v20.4s, v10.4s, v2.s[2]     \n"// sum2 += (a01-a71) * k21
								"fmla    v21.4s, v11.4s, v2.s[2]     \n"//
								"fmla    v22.4s, v10.4s, v2.s[3]     \n"// sum3 += (a01-a71) * k31
								"fmla    v23.4s, v11.4s, v2.s[3]     \n"//
								"fmla    v24.4s, v10.4s, v3.s[0]     \n"// sum4 += (a01-a71) * k41
								"fmla    v25.4s, v11.4s, v3.s[0]     \n"//
								"fmla    v26.4s, v10.4s, v3.s[1]     \n"// sum5 += (a01-a71) * k51
								"fmla    v27.4s, v11.4s, v3.s[1]     \n"//
								"fmla    v28.4s, v10.4s, v3.s[2]     \n"// sum6 += (a01-a71) * k61
								"fmla    v29.4s, v11.4s, v3.s[2]     \n"//
								"fmla    v30.4s, v10.4s, v3.s[3]     \n"// sum7 += (a01-a71) * k71
								"fmla    v31.4s, v11.4s, v3.s[3]     \n"//
																		// k2
								"fmla    v16.4s, v12.4s, v4.s[0]     \n"// sum0 += (a02-a72) * k02
								"fmla    v17.4s, v13.4s, v4.s[0]     \n"//
								"fmla    v18.4s, v12.4s, v4.s[1]     \n"// sum1 += (a02-a72) * k12
								"fmla    v19.4s, v13.4s, v4.s[1]     \n"//
								"fmla    v20.4s, v12.4s, v4.s[2]     \n"// sum2 += (a02-a72) * k22
								"fmla    v21.4s, v13.4s, v4.s[2]     \n"//
								"fmla    v22.4s, v12.4s, v4.s[3]     \n"// sum3 += (a02-a72) * k32
								"fmla    v23.4s, v13.4s, v4.s[3]     \n"//
								"fmla    v24.4s, v12.4s, v5.s[0]     \n"// sum4 += (a02-a72) * k42
								"fmla    v25.4s, v13.4s, v5.s[0]     \n"//
								"fmla    v26.4s, v12.4s, v5.s[1]     \n"// sum5 += (a02-a72) * k52
								"fmla    v27.4s, v13.4s, v5.s[1]     \n"//
								"fmla    v28.4s, v12.4s, v5.s[2]     \n"// sum6 += (a02-a72) * k62
								"fmla    v29.4s, v13.4s, v5.s[2]     \n"//
								"fmla    v30.4s, v12.4s, v5.s[3]     \n"// sum7 += (a02-a72) * k72
								"fmla    v31.4s, v13.4s, v5.s[3]     \n"//
																		// k3
								"fmla    v16.4s, v14.4s, v6.s[0]     \n"// sum0 += (a03-a73) * k03
								"fmla    v17.4s, v15.4s, v6.s[0]     \n"//
								"fmla    v18.4s, v14.4s, v6.s[1]     \n"// sum1 += (a03-a73) * k13
								"fmla    v19.4s, v15.4s, v6.s[1]     \n"//
								"fmla    v20.4s, v14.4s, v6.s[2]     \n"// sum2 += (a03-a73) * k23
								"fmla    v21.4s, v15.4s, v6.s[2]     \n"//
								"fmla    v22.4s, v14.4s, v6.s[3]     \n"// sum3 += (a03-a73) * k33
								"fmla    v23.4s, v15.4s, v6.s[3]     \n"//
								"fmla    v24.4s, v14.4s, v7.s[0]     \n"// sum4 += (a03-a73) * k43
								"fmla    v25.4s, v15.4s, v7.s[0]     \n"//
								"fmla    v26.4s, v14.4s, v7.s[1]     \n"// sum5 += (a03-a73) * k53
								"fmla    v27.4s, v15.4s, v7.s[1]     \n"//
								"fmla    v28.4s, v14.4s, v7.s[2]     \n"// sum6 += (a03-a73) * k63
								"fmla    v29.4s, v15.4s, v7.s[2]     \n"//
								"fmla    v30.4s, v14.4s, v7.s[3]     \n"// sum7 += (a03-a73) * k73
								"fmla    v31.4s, v15.4s, v7.s[3]     \n"//

								"subs   w4, w4, #1                   \n"
								"bne    0b                           \n"

								"1:                                  \n"

								// remain loop
								"and    w4, %w20, #3                 \n"// w4 = remain = inch & 3;
								"cmp    w4, #0                       \n"
								"beq    3f                           \n"

								"2:                                  \n"

								"prfm   pldl1keep, [%9, #256]        \n"
								"ld1    {v0.4s, v1.4s}, [%9], #32    \n"

								"prfm   pldl1keep, [%8, #256]        \n"
								"ld1    {v8.4s, v9.4s}, [%8], #32    \n"

								// k0
								"fmla    v16.4s, v8.4s, v0.s[0]      \n"// sum0 += (a00-a70) * k00
								"fmla    v17.4s, v9.4s, v0.s[0]      \n"//
								"fmla    v18.4s, v8.4s, v0.s[1]      \n"// sum1 += (a00-a70) * k10
								"fmla    v19.4s, v9.4s, v0.s[1]      \n"//
								"fmla    v20.4s, v8.4s, v0.s[2]      \n"// sum2 += (a00-a70) * k20
								"fmla    v21.4s, v9.4s, v0.s[2]      \n"//
								"fmla    v22.4s, v8.4s, v0.s[3]      \n"// sum3 += (a00-a70) * k30
								"fmla    v23.4s, v9.4s, v0.s[3]      \n"//
								"fmla    v24.4s, v8.4s, v1.s[0]      \n"// sum4 += (a00-a70) * k40
								"fmla    v25.4s, v9.4s, v1.s[0]      \n"//
								"fmla    v26.4s, v8.4s, v1.s[1]      \n"// sum5 += (a00-a70) * k50
								"fmla    v27.4s, v9.4s, v1.s[1]      \n"//
								"fmla    v28.4s, v8.4s, v1.s[2]      \n"// sum6 += (a00-a70) * k60
								"fmla    v29.4s, v9.4s, v1.s[2]      \n"//
								"fmla    v30.4s, v8.4s, v1.s[3]      \n"// sum7 += (a00-a70) * k70
								"fmla    v31.4s, v9.4s, v1.s[3]      \n"//

								"subs   w4, w4, #1                   \n"

								"bne    2b                           \n"

								"3:                                  \n"

								"st1    {v16.4s, v17.4s}, [%0]       \n"
								"st1    {v18.4s, v19.4s}, [%1]       \n"
								"st1    {v20.4s, v21.4s}, [%2]       \n"
								"st1    {v22.4s, v23.4s}, [%3]       \n"
								"st1    {v24.4s, v25.4s}, [%4]       \n"
								"st1    {v26.4s, v27.4s}, [%5]       \n"
								"st1    {v28.4s, v29.4s}, [%6]       \n"
								"st1    {v30.4s, v31.4s}, [%7]       \n"

								: "=r"(output0), // %0
								"=r"(output1), // %1
								"=r"(output2), // %2
								"=r"(output3), // %3
								"=r"(output4), // %4
								"=r"(output5), // %5
								"=r"(output6), // %6
								"=r"(output7), // %7
								"=r"(vb),      // %8
								"=r"(va)       // %9
								: "0"(output0),
								"1"(output1),
								"2"(output2),
								"3"(output3),
								"4"(output4),
								"5"(output5),
								"6"(output6),
								"7"(output7),
								"8"(vb),
								"9"(va),
								"r"(L),        // %20
								"r"(biasptr)   // %21
								: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
								);
#else                
							float sum0[8] = { 0 };
							float sum1[8] = { 0 };
							float sum2[8] = { 0 };
							float sum3[8] = { 0 };
							float sum4[8] = { 0 };
							float sum5[8] = { 0 };
							float sum6[8] = { 0 };
							float sum7[8] = { 0 };

							int k = 0;
							for (; k + 7 < L; k = k + 8)
							{
								for (int n = 0; n < 8; n++)
								{
									sum0[n] += va[0] * vb[n];
									sum1[n] += va[1] * vb[n];
									sum2[n] += va[2] * vb[n];
									sum3[n] += va[3] * vb[n];
									sum4[n] += va[4] * vb[n];
									sum5[n] += va[5] * vb[n];
									sum6[n] += va[6] * vb[n];
									sum7[n] += va[7] * vb[n];
									va += 8;

									sum0[n] += va[0] * vb[n + 8];
									sum1[n] += va[1] * vb[n + 8];
									sum2[n] += va[2] * vb[n + 8];
									sum3[n] += va[3] * vb[n + 8];
									sum4[n] += va[4] * vb[n + 8];
									sum5[n] += va[5] * vb[n + 8];
									sum6[n] += va[6] * vb[n + 8];
									sum7[n] += va[7] * vb[n + 8];
									va += 8;

									sum0[n] += va[0] * vb[n + 16];
									sum1[n] += va[1] * vb[n + 16];
									sum2[n] += va[2] * vb[n + 16];
									sum3[n] += va[3] * vb[n + 16];
									sum4[n] += va[4] * vb[n + 16];
									sum5[n] += va[5] * vb[n + 16];
									sum6[n] += va[6] * vb[n + 16];
									sum7[n] += va[7] * vb[n + 16];
									va += 8;

									sum0[n] += va[0] * vb[n + 24];
									sum1[n] += va[1] * vb[n + 24];
									sum2[n] += va[2] * vb[n + 24];
									sum3[n] += va[3] * vb[n + 24];
									sum4[n] += va[4] * vb[n + 24];
									sum5[n] += va[5] * vb[n + 24];
									sum6[n] += va[6] * vb[n + 24];
									sum7[n] += va[7] * vb[n + 24];
									va += 8;

									sum0[n] += va[0] * vb[n + 32];
									sum1[n] += va[1] * vb[n + 32];
									sum2[n] += va[2] * vb[n + 32];
									sum3[n] += va[3] * vb[n + 32];
									sum4[n] += va[4] * vb[n + 32];
									sum5[n] += va[5] * vb[n + 32];
									sum6[n] += va[6] * vb[n + 32];
									sum7[n] += va[7] * vb[n + 32];
									va += 8;

									sum0[n] += va[0] * vb[n + 40];
									sum1[n] += va[1] * vb[n + 40];
									sum2[n] += va[2] * vb[n + 40];
									sum3[n] += va[3] * vb[n + 40];
									sum4[n] += va[4] * vb[n + 40];
									sum5[n] += va[5] * vb[n + 40];
									sum6[n] += va[6] * vb[n + 40];
									sum7[n] += va[7] * vb[n + 40];
									va += 8;

									sum0[n] += va[0] * vb[n + 48];
									sum1[n] += va[1] * vb[n + 48];
									sum2[n] += va[2] * vb[n + 48];
									sum3[n] += va[3] * vb[n + 48];
									sum4[n] += va[4] * vb[n + 48];
									sum5[n] += va[5] * vb[n + 48];
									sum6[n] += va[6] * vb[n + 48];
									sum7[n] += va[7] * vb[n + 48];
									va += 8;

									sum0[n] += va[0] * vb[n + 56];
									sum1[n] += va[1] * vb[n + 56];
									sum2[n] += va[2] * vb[n + 56];
									sum3[n] += va[3] * vb[n + 56];
									sum4[n] += va[4] * vb[n + 56];
									sum5[n] += va[5] * vb[n + 56];
									sum6[n] += va[6] * vb[n + 56];
									sum7[n] += va[7] * vb[n + 56];
									va -= 56;
								}

								va += 64;
								vb += 64;
							}

							for (; k < L; k++)
							{
								for (int n = 0; n < 8; n++)
								{
									sum0[n] += va[0] * vb[n];
									sum1[n] += va[1] * vb[n];
									sum2[n] += va[2] * vb[n];
									sum3[n] += va[3] * vb[n];
									sum4[n] += va[4] * vb[n];
									sum5[n] += va[5] * vb[n];
									sum6[n] += va[6] * vb[n];
									sum7[n] += va[7] * vb[n];
								}

								va += 8;
								vb += 8;
							}

							for (int n = 0; n < 8; n++)
							{
								output0[n] = sum0[n] + biasptr[0];
								output1[n] = sum1[n] + biasptr[1];
								output2[n] = sum2[n] + biasptr[2];
								output3[n] = sum3[n] + biasptr[3];
								output4[n] = sum4[n] + biasptr[4];
								output5[n] = sum5[n] + biasptr[5];
								output6[n] = sum6[n] + biasptr[6];
								output7[n] = sum7[n] + biasptr[7];
							}
#endif // __ARM_NEON
							output0 += 8;
							output1 += 8;
							output2 += 8;
							output3 += 8;
							output4 += 8;
							output5 += 8;
							output6 += 8;
							output7 += 8;
						}

						for (; j < N; j++)
						{
							const float* vb = bottom_tm_data + (j / 8 + j % 8) * bottom_tm_cstep;
							const float* va = kernel_tm_gemm_data + (i / 8) * kernel_tm_gemm_cstep;

#if __ARM_NEON
							asm volatile(
								"ld1    {v14.4s, v15.4s}, [%21]      \n" // sum0_7 inital with bias
								"eor    v16.16b, v16.16b, v16.16b    \n" // sum0
								"eor    v17.16b, v17.16b, v17.16b    \n" // sum1
								"eor    v18.16b, v18.16b, v18.16b    \n" // sum2
								"eor    v19.16b, v19.16b, v19.16b    \n" // sum3
								"eor    v20.16b, v20.16b, v20.16b    \n" // sum4
								"eor    v21.16b, v21.16b, v21.16b    \n" // sum5
								"eor    v22.16b, v22.16b, v22.16b    \n" // sum6
								"eor    v23.16b, v23.16b, v23.16b    \n" // sum7

								"lsr         w4, %w20, #2            \n"// r4 = nn = L >> 2
								"cmp         w4, #0                  \n"
								"beq         1f                      \n"

								"0:                                  \n"// for (; k+3<L; k=k+4)

								"prfm   pldl1keep, [%9, #256]                       \n"
								"ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64     \n" // k
								"ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%9], #64     \n"

								"prfm   pldl1keep, [%8, #128]        \n"
								"ld1    {v8.4s}, [%8], #16           \n" // d

																		 // k0
								"fmla    v16.4s, v0.4s, v8.s[0]      \n"// sum0 += (k00-k70) * a00
								"fmla    v17.4s, v1.4s, v8.s[0]      \n"//
								"fmla    v18.4s, v2.4s, v8.s[1]      \n"// sum1 += (k01-k71) * a10
								"fmla    v19.4s, v3.4s, v8.s[1]      \n"//
								"fmla    v20.4s, v4.4s, v8.s[2]      \n"// sum2 += (k02-k72) * a20
								"fmla    v21.4s, v5.4s, v8.s[2]      \n"//
								"fmla    v22.4s, v6.4s, v8.s[3]      \n"// sum3 += (k03-k73) * a30
								"fmla    v23.4s, v7.4s, v8.s[3]      \n"//

								"subs   w4, w4, #1                   \n"
								"bne    0b                           \n"

								"fadd   v16.4s, v16.4s, v18.4s       \n"
								"fadd   v17.4s, v17.4s, v19.4s       \n"
								"fadd   v20.4s, v20.4s, v22.4s       \n"
								"fadd   v21.4s, v21.4s, v23.4s       \n"
								"fadd   v16.4s, v16.4s, v20.4s       \n"
								"fadd   v17.4s, v17.4s, v21.4s       \n"
								"fadd   v14.4s, v14.4s, v16.4s       \n"
								"fadd   v15.4s, v15.4s, v17.4s       \n"

								"1:                                  \n"

								// remain loop
								"and    w4, %w20, #3                 \n"// w4 = remain = inch & 3;
								"cmp    w4, #0                       \n"
								"beq    3f                           \n"

								"2:                                  \n"

								"prfm   pldl1keep, [%9, #256]        \n"
								"ld1    {v0.4s, v1.4s}, [%9], #32    \n"
								"prfm   pldl1keep, [%8, #32]         \n"
								"ld1r   {v8.4s}, [%8], #4            \n"

								// k0
								"fmla   v14.4s, v8.4s, v0.4s         \n"// sum0 += (k00-k70) * a00
								"fmla   v15.4s, v8.4s, v1.4s         \n"//

								"subs   w4, w4, #1                   \n"

								"bne    2b                           \n"

								"3:                                  \n"

								"st1    {v14.s}[0], [%0]             \n"
								"st1    {v14.s}[1], [%1]             \n"
								"st1    {v14.s}[2], [%2]             \n"
								"st1    {v14.s}[3], [%3]             \n"
								"st1    {v15.s}[0], [%4]             \n"
								"st1    {v15.s}[1], [%5]             \n"
								"st1    {v15.s}[2], [%6]             \n"
								"st1    {v15.s}[3], [%7]             \n"

								: "=r"(output0), // %0
								"=r"(output1), // %1
								"=r"(output2), // %2
								"=r"(output3), // %3
								"=r"(output4), // %4
								"=r"(output5), // %5
								"=r"(output6), // %6
								"=r"(output7), // %7
								"=r"(vb),      // %8
								"=r"(va)       // %9
								: "0"(output0),
								"1"(output1),
								"2"(output2),
								"3"(output3),
								"4"(output4),
								"5"(output5),
								"6"(output6),
								"7"(output7),
								"8"(vb),
								"9"(va),
								"r"(L),        // %20 
								"r"(biasptr)   // %21
								: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
								);
#else
							float sum0 = biasptr[0];
							float sum1 = biasptr[1];
							float sum2 = biasptr[2];
							float sum3 = biasptr[3];
							float sum4 = biasptr[4];
							float sum5 = biasptr[5];
							float sum6 = biasptr[6];
							float sum7 = biasptr[7];

							for (int k = 0; k < L; k++)
							{
								sum0 += va[0] * vb[0];
								sum1 += va[1] * vb[0];
								sum2 += va[2] * vb[0];
								sum3 += va[3] * vb[0];
								sum4 += va[4] * vb[0];
								sum5 += va[5] * vb[0];
								sum6 += va[6] * vb[0];
								sum7 += va[7] * vb[0];

								va += 8;
								vb += 1;
							}

							output0[0] = sum0;
							output1[0] = sum1;
							output2[0] = sum2;
							output3[0] = sum3;
							output4[0] = sum4;
							output5[0] = sum5;
							output6[0] = sum6;
							output7[0] = sum7;
#endif // __ARM_NEON
							output0++;
							output1++;
							output2++;
							output3++;
							output4++;
							output5++;
							output6++;
							output7++;
						}
					}
#endif // __aarch64__

					nn_outch = (outch - remain_outch_start) >> 2;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
					for (int pp = 0; pp < nn_outch; pp++)
					{
						int i = remain_outch_start + pp * 4;

						float* output0 = top_data + (i)* top_cstep;
						float* output1 = top_data + (i + 1) * top_cstep;
						float* output2 = top_data + (i + 2) * top_cstep;
						float* output3 = top_data + (i + 3) * top_cstep;

						const float zeros[4] = { 0.f, 0.f, 0.f, 0.f };
						const float* biasptr = this->bias_term_ ? bias + i : zeros;

						int j = 0;
						for (; j + 7 < N; j = j + 8)
						{
							const float* vb = bottom_tm_data + (j / 8) * bottom_tm_cstep;
#if __ARM_NEON && __aarch64__
							const float* va = kernel_tm_gemm_data + (i / 8 + (i % 8) / 4) * kernel_tm_gemm_cstep;
#else                
							const float* va = kernel_tm_gemm_data + (i / 4) * kernel_tm_gemm_cstep;
#endif // __ARM_NEON && __aarch64__

#if __ARM_NEON
#if __aarch64__
							asm volatile(
								"ld1    {v0.4s}, [%13]               \n"
								"dup    v16.4s, v0.s[0]              \n"// sum0
								"dup    v17.4s, v0.s[0]              \n"
								"dup    v18.4s, v0.s[1]              \n"// sum1
								"dup    v19.4s, v0.s[1]              \n"
								"dup    v20.4s, v0.s[2]              \n"// sum2
								"dup    v21.4s, v0.s[2]              \n"
								"dup    v22.4s, v0.s[3]              \n"// sum3
								"dup    v23.4s, v0.s[3]              \n"

								"lsr         w4, %w12, #2            \n"// r4 = nn = L >> 2
								"cmp         w4, #0                  \n"
								"beq         1f                      \n"

								"0:                                  \n"// for (; k+3<L; k=k+4)

								"prfm   pldl1keep, [%5, #512]                       \n"
								"ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64     \n" // kernel

								"prfm   pldl1keep, [%4, #512]                       \n"
								"ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%4], #64   \n" // data
								"ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%4], #64 \n"

								"subs   w4, w4, #1                   \n"
								// k0
								"fmla    v16.4s, v8.4s, v0.s[0]      \n"// sum0 += (a00-a70) * k00
								"fmla    v17.4s, v9.4s, v0.s[0]      \n"//
								"fmla    v18.4s, v8.4s, v0.s[1]      \n"// sum1 += (a00-a70) * k10
								"fmla    v19.4s, v9.4s, v0.s[1]      \n"//
								"fmla    v20.4s, v8.4s, v0.s[2]      \n"// sum2 += (a00-a70) * k20
								"fmla    v21.4s, v9.4s, v0.s[2]      \n"//
								"fmla    v22.4s, v8.4s, v0.s[3]      \n"// sum3 += (a00-a70) * k30
								"fmla    v23.4s, v9.4s, v0.s[3]      \n"//
																		// k1
								"fmla    v16.4s, v10.4s, v1.s[0]     \n"// sum0 += (a01-a71) * k01
								"fmla    v17.4s, v11.4s, v1.s[0]     \n"//
								"fmla    v18.4s, v10.4s, v1.s[1]     \n"// sum1 += (a01-a71) * k11
								"fmla    v19.4s, v11.4s, v1.s[1]     \n"//
								"fmla    v20.4s, v10.4s, v1.s[2]     \n"// sum2 += (a01-a71) * k21
								"fmla    v21.4s, v11.4s, v1.s[2]     \n"//
								"fmla    v22.4s, v10.4s, v1.s[3]     \n"// sum3 += (a01-a71) * k31
								"fmla    v23.4s, v11.4s, v1.s[3]     \n"//
																		// k2
								"fmla    v16.4s, v12.4s, v2.s[0]     \n"// sum0 += (a02-a72) * k02
								"fmla    v17.4s, v13.4s, v2.s[0]     \n"//
								"fmla    v18.4s, v12.4s, v2.s[1]     \n"// sum1 += (a02-a72) * k12
								"fmla    v19.4s, v13.4s, v2.s[1]     \n"//
								"fmla    v20.4s, v12.4s, v2.s[2]     \n"// sum2 += (a02-a72) * k22
								"fmla    v21.4s, v13.4s, v2.s[2]     \n"//
								"fmla    v22.4s, v12.4s, v2.s[3]     \n"// sum3 += (a02-a72) * k32
								"fmla    v23.4s, v13.4s, v2.s[3]     \n"//
																		// k3
								"fmla    v16.4s, v14.4s, v3.s[0]     \n"// sum0 += (a03-a73) * k03
								"fmla    v17.4s, v15.4s, v3.s[0]     \n"//
								"fmla    v18.4s, v14.4s, v3.s[1]     \n"// sum1 += (a03-a73) * k13
								"fmla    v19.4s, v15.4s, v3.s[1]     \n"//
								"fmla    v20.4s, v14.4s, v3.s[2]     \n"// sum2 += (a03-a73) * k23
								"fmla    v21.4s, v15.4s, v3.s[2]     \n"//
								"fmla    v22.4s, v14.4s, v3.s[3]     \n"// sum3 += (a03-a73) * k33
								"fmla    v23.4s, v15.4s, v3.s[3]     \n"//

								"bne    0b                           \n"

								"1:                                  \n"

								// remain loop
								"and    w4, %w12, #3                 \n"// w4 = remain = inch & 3;
								"cmp    w4, #0                       \n"
								"beq    3f                           \n"

								"2:                                  \n"

								"prfm   pldl1keep, [%5, #256]        \n"
								"ld1    {v0.4s}, [%5], #16           \n"
								"prfm   pldl1keep, [%4, #256]        \n"
								"ld1    {v8.4s, v9.4s}, [%4], #32    \n"
								// k0
								"fmla    v16.4s, v8.4s, v0.s[0]      \n"// sum0 += (a00-a70) * k00
								"fmla    v17.4s, v9.4s, v0.s[0]      \n"//
								"fmla    v18.4s, v8.4s, v0.s[1]      \n"// sum1 += (a00-a70) * k10
								"fmla    v19.4s, v9.4s, v0.s[1]      \n"//
								"fmla    v20.4s, v8.4s, v0.s[2]      \n"// sum2 += (a00-a70) * k20
								"fmla    v21.4s, v9.4s, v0.s[2]      \n"//
								"fmla    v22.4s, v8.4s, v0.s[3]      \n"// sum3 += (a00-a70) * k30
								"fmla    v23.4s, v9.4s, v0.s[3]      \n"//

								"subs   w4, w4, #1                   \n"

								"bne    2b                           \n"

								"3:                                  \n"

								"st1    {v16.4s, v17.4s}, [%0]       \n"
								"st1    {v18.4s, v19.4s}, [%1]       \n"
								"st1    {v20.4s, v21.4s}, [%2]       \n"
								"st1    {v22.4s, v23.4s}, [%3]       \n"

								: "=r"(output0), // %0
								"=r"(output1), // %1
								"=r"(output2), // %2
								"=r"(output3), // %3
								"=r"(vb),      // %4
								"=r"(va)       // %5
								: "0"(output0),
								"1"(output1),
								"2"(output2),
								"3"(output3),
								"4"(vb),
								"5"(va),
								"r"(L),        // %12 
								"r"(biasptr)   // %13
								: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
								);
#else
							asm volatile(
								"vld1.f32   {d0-d1}, [%13]      \n"
								"vdup.f32   q8, d0[0]           \n"
								"vdup.f32   q9, d0[0]           \n"
								"vdup.f32   q10, d0[1]          \n"
								"vdup.f32   q11, d0[1]          \n"
								"vdup.f32   q12, d1[0]          \n"
								"vdup.f32   q13, d1[0]          \n"
								"vdup.f32   q14, d1[1]          \n"
								"vdup.f32   q15, d1[1]          \n"

								"lsr         r4, %12, #2        \n"// r4 = nn = L >> 2
								"cmp         r4, #0             \n"
								"beq         1f                 \n"

								"0:                             \n"// for(; nn != 0; nn--)
								"pld        [%5, #512]          \n"
								"vldm       %5!, {d0-d7}        \n"// kernel
								"pld        [%4, #512]          \n"
								"vldm       %4!, {d8-d15}       \n"// data

								"vmla.f32   q8, q4, d0[0]       \n"// sum0 = (a00-a07) * k00
								"vmla.f32   q9, q5, d0[0]       \n"
								"vmla.f32   q10, q4, d0[1]      \n"// sum1 = (a00-a07) * k10
								"vmla.f32   q11, q5, d0[1]      \n"
								"vmla.f32   q12, q4, d1[0]      \n"// sum2 = (a00-a07) * k20
								"vmla.f32   q13, q5, d1[0]      \n"
								"vmla.f32   q14, q4, d1[1]      \n"// sum3 = (a00-a07) * k30
								"vmla.f32   q15, q5, d1[1]      \n"

								"vmla.f32   q8, q6, d2[0]       \n"// sum0 += (a10-a17) * k01
								"vmla.f32   q9, q7, d2[0]       \n"
								"vmla.f32   q10, q6, d2[1]      \n"// sum1 += (a10-a17) * k11
								"vmla.f32   q11, q7, d2[1]      \n"
								"vmla.f32   q12, q6, d3[0]      \n"// sum2 += (a10-a17) * k21
								"vmla.f32   q13, q7, d3[0]      \n"
								"vmla.f32   q14, q6, d3[1]      \n"// sum3 += (a10-a17) * k31
								"vmla.f32   q15, q7, d3[1]      \n"

								"pld        [%4, #512]          \n"
								"vldm       %4!, {d8-d15}       \n"// data

								"vmla.f32   q8, q4, d4[0]       \n"// sum0 += (a20-a27) * k02
								"vmla.f32   q9, q5, d4[0]       \n"
								"vmla.f32   q10, q4, d4[1]      \n"// sum1 += (a20-a27) * k12
								"vmla.f32   q11, q5, d4[1]      \n"
								"vmla.f32   q12, q4, d5[0]      \n"// sum2 += (a20-a27) * k22
								"vmla.f32   q13, q5, d5[0]      \n"
								"vmla.f32   q14, q4, d5[1]      \n"// sum3 += (a20-a27) * k32
								"vmla.f32   q15, q5, d5[1]      \n"

								"vmla.f32   q8, q6, d6[0]       \n"// sum0 += (a30-a37) * k03
								"vmla.f32   q9, q7, d6[0]       \n"
								"vmla.f32   q10, q6, d6[1]      \n"// sum1 += (a30-a37) * k13
								"vmla.f32   q11, q7, d6[1]      \n"
								"vmla.f32   q12, q6, d7[0]      \n"// sum2 += (a30-a37) * k23
								"vmla.f32   q13, q7, d7[0]      \n"
								"vmla.f32   q14, q6, d7[1]      \n"// sum3 += (a30-a37) * k33
								"vmla.f32   q15, q7, d7[1]      \n"

								"subs        r4, r4, #1         \n"
								"bne         0b                 \n"// end for

								"1:                             \n"
								// remain loop
								"and         r4, %12, #3        \n"// r4 = remain = inch & 3
								"cmp         r4, #0             \n"
								"beq         3f                 \n"

								"2:                             \n"// for(; remain != 0; remain--)

								"pld        [%5, #128]          \n"
								"vld1.f32   {d0-d1}, [%5]!      \n"
								"pld        [%4, #256]          \n"
								"vld1.f32   {d8-d11}, [%4]!     \n"

								"vmla.f32   q8, q4, d0[0]       \n"// sum0 += (a00-a70) * k00
								"vmla.f32   q9, q5, d0[0]       \n"
								"vmla.f32   q10, q4, d0[1]      \n"// sum1 += (a00-a70) * k10
								"vmla.f32   q11, q5, d0[1]      \n"
								"vmla.f32   q12, q4, d1[0]      \n"// sum2 += (a00-a70) * k20
								"vmla.f32   q13, q5, d1[0]      \n"
								"vmla.f32   q14, q4, d1[1]      \n"// sum3 += (a00-a70) * k30
								"vmla.f32   q15, q5, d1[1]      \n"

								"subs        r4, r4, #1         \n"
								"bne         2b                 \n"

								"3:                             \n"// store the result to memory
								"vst1.f32    {d16-d19}, [%0]    \n"
								"vst1.f32    {d20-d23}, [%1]    \n"
								"vst1.f32    {d24-d27}, [%2]    \n"
								"vst1.f32    {d28-d31}, [%3]    \n"

								: "=r"(output0), // %0
								"=r"(output1), // %1
								"=r"(output2), // %2
								"=r"(output3), // %3
								"=r"(vb),      // %4
								"=r"(va)       // %5
								: "0"(output0),
								"1"(output1),
								"2"(output2),
								"3"(output3),
								"4"(vb),
								"5"(va),
								"r"(L),        // %12 
								"r"(biasptr)   // %13
								: "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
								);
#endif // __aarch64__
#else
							float sum0[8] = { 0 };
							float sum1[8] = { 0 };
							float sum2[8] = { 0 };
							float sum3[8] = { 0 };

							int k = 0;
							for (; k + 7 < L; k = k + 8)
							{
								for (int n = 0; n < 8; n++)
								{
									sum0[n] += va[0] * vb[n];
									sum1[n] += va[1] * vb[n];
									sum2[n] += va[2] * vb[n];
									sum3[n] += va[3] * vb[n];
									va += 4;

									sum0[n] += va[0] * vb[n + 8];
									sum1[n] += va[1] * vb[n + 8];
									sum2[n] += va[2] * vb[n + 8];
									sum3[n] += va[3] * vb[n + 8];
									va += 4;

									sum0[n] += va[0] * vb[n + 16];
									sum1[n] += va[1] * vb[n + 16];
									sum2[n] += va[2] * vb[n + 16];
									sum3[n] += va[3] * vb[n + 16];
									va += 4;

									sum0[n] += va[0] * vb[n + 24];
									sum1[n] += va[1] * vb[n + 24];
									sum2[n] += va[2] * vb[n + 24];
									sum3[n] += va[3] * vb[n + 24];
									va += 4;

									sum0[n] += va[0] * vb[n + 32];
									sum1[n] += va[1] * vb[n + 32];
									sum2[n] += va[2] * vb[n + 32];
									sum3[n] += va[3] * vb[n + 32];
									va += 4;

									sum0[n] += va[0] * vb[n + 40];
									sum1[n] += va[1] * vb[n + 40];
									sum2[n] += va[2] * vb[n + 40];
									sum3[n] += va[3] * vb[n + 40];
									va += 4;

									sum0[n] += va[0] * vb[n + 48];
									sum1[n] += va[1] * vb[n + 48];
									sum2[n] += va[2] * vb[n + 48];
									sum3[n] += va[3] * vb[n + 48];
									va += 4;

									sum0[n] += va[0] * vb[n + 56];
									sum1[n] += va[1] * vb[n + 56];
									sum2[n] += va[2] * vb[n + 56];
									sum3[n] += va[3] * vb[n + 56];
									va -= 28;
								}

								va += 32;
								vb += 64;
							}

							for (; k < L; k++)
							{
								for (int n = 0; n < 8; n++)
								{
									sum0[n] += va[0] * vb[n];
									sum1[n] += va[1] * vb[n];
									sum2[n] += va[2] * vb[n];
									sum3[n] += va[3] * vb[n];
								}

								va += 4;
								vb += 8;
							}

							for (int n = 0; n < 8; n++)
							{
								output0[n] = sum0[n] + biasptr[0];
								output1[n] = sum1[n] + biasptr[1];
								output2[n] = sum2[n] + biasptr[2];
								output3[n] = sum3[n] + biasptr[3];
							}
#endif // __ARM_NEON
							output0 += 8;
							output1 += 8;
							output2 += 8;
							output3 += 8;
						}

						for (; j < N; j++)
						{
							float* vb = bottom_tm_data + (j / 8 + j % 8) * bottom_tm_cstep;
#if __ARM_NEON && __aarch64__
							const float* va = kernel_tm_gemm_data + (i / 8 + (i % 8) / 4) * kernel_tm_gemm_cstep;
#else                
							const float* va = kernel_tm_gemm_data + (i / 4) * kernel_tm_gemm_cstep;
#endif // __ARM_NEON && __aarch64__

#if __ARM_NEON
#if __aarch64__
							asm volatile(
								"ld1    {v14.4s}, [%13]              \n" // sum0_3 inital with bias

								"lsr         w4, %w12, #2            \n"// r4 = nn = L >> 2
								"cmp         w4, #0                  \n"
								"beq         1f                      \n"

								"eor    v16.16b, v16.16b, v16.16b    \n" // sum0
								"eor    v17.16b, v17.16b, v17.16b    \n" // sum1
								"eor    v18.16b, v18.16b, v18.16b    \n" // sum2
								"eor    v19.16b, v19.16b, v19.16b    \n" // sum3                    

								"0:                                  \n"// for (; k+3<L; k=k+4)

								"prfm   pldl1keep, [%5, #256]                       \n"
								"ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64     \n" // k

								"prfm   pldl1keep, [%4, #128]        \n"
								"ld1    {v8.4s}, [%4], #16           \n" // d

								"subs   w4, w4, #1                   \n"
								"fmla    v16.4s, v0.4s, v8.s[0]      \n"// sum0 += (k00-k30) * a00
								"fmla    v17.4s, v1.4s, v8.s[1]      \n"// sum1 += (k01-k31) * a10
								"fmla    v18.4s, v2.4s, v8.s[2]      \n"// sum2 += (k02-k32) * a20
								"fmla    v19.4s, v3.4s, v8.s[3]      \n"// sum3 += (k03-k33) * a30

								"bne    0b                           \n"

								"add      v16.4s, v16.4s, v18.4s     \n"
								"add      v17.4s, v17.4s, v19.4s     \n"
								"add      v14.4s, v16.4s, v17.4s     \n"

								"1:                                  \n"

								// remain loop
								"and    w4, %w12, #3                 \n"// w4 = remain = inch & 3;
								"cmp    w4, #0                       \n"
								"beq    3f                           \n"

								"2:                                  \n"

								"prfm   pldl1keep, [%5, #128]        \n"
								"ld1    {v0.4s}, [%5], #16           \n"
								"prfm   pldl1keep, [%4, #32]         \n"
								"ld1r   {v8.4s}, [%4], #4            \n"

								"subs   w4, w4, #1                   \n"
								// k0
								"fmla   v14.4s, v8.4s, v0.4s         \n"// sum0 += (k00-k30) * a00
								"bne    2b                           \n"

								"3:                                  \n"

								"st1    {v14.s}[0], [%0]             \n"
								"st1    {v14.s}[1], [%1]             \n"
								"st1    {v14.s}[2], [%2]             \n"
								"st1    {v14.s}[3], [%3]             \n"

								: "=r"(output0), // %0
								"=r"(output1), // %1
								"=r"(output2), // %2
								"=r"(output3), // %3
								"=r"(vb),      // %4
								"=r"(va)       // %5
								: "0"(output0),
								"1"(output1),
								"2"(output2),
								"3"(output3),
								"4"(vb),
								"5"(va),
								"r"(L),        // %12 
								"r"(biasptr)   // %13
								: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19"
								);
#else
							asm volatile(
								// inch loop
								"vld1.f32   {d24-d25}, [%13]    \n"

								"lsr         r4, %12, #2        \n"// r4 = nn = L >> 2
								"cmp         r4, #0             \n"
								"beq         1f                 \n"

								"veor       q8, q8, q8          \n"
								"veor       q9, q9, q9          \n"
								"veor       q10, q10, q10       \n"
								"veor       q11, q11, q11       \n"

								"0:                             \n"// for(; nn != 0; nn--)
								"pld        [%5, #512]          \n"
								"vldm       %5!, {d0-d7}        \n"// kernel
								"pld        [%4, #128]          \n"
								"vld1.f32   {d8-d9}, [%4]!      \n"// data

								"vmla.f32   q8, q0, d8[0]       \n"// (k00-k30) * a00
								"vmla.f32   q9, q1, d8[1]       \n"// (k01-k31) * a01
								"vmla.f32   q10, q2, d9[0]      \n"// (k02-k32) * a02
								"vmla.f32   q11, q3, d9[1]      \n"// (k03-k33) * a03

								"subs        r4, r4, #1         \n"
								"bne         0b                 \n"// end for

								"vadd.f32   q8, q8, q9          \n"
								"vadd.f32   q10, q10, q11       \n"
								"vadd.f32   q8, q8, q10         \n"
								"vadd.f32   q12, q12, q8        \n"

								"1:                             \n"
								// remain loop
								"and         r4, %12, #3        \n"// r4 = remain = inch & 3
								"cmp         r4, #0             \n"
								"beq         3f                 \n"

								"2:                             \n"// for(; remain != 0; remain--)
								"pld        [%5, #128]          \n"
								"vld1.f32   {d0-d1}, [%5]!      \n"
								"pld        [%4, #32]           \n"
								"vld1.f32   {d8[],d9[]}, [%4]!  \n"

								"subs       r4, r4, #1          \n"

								"vmla.f32   q12, q0, q4         \n"
								"bne         2b                 \n"

								"3:                             \n"// store the result to memory
								"vst1.f32    {d24[0]}, [%0]     \n"
								"vst1.f32    {d24[1]}, [%1]     \n"
								"vst1.f32    {d25[0]}, [%2]     \n"
								"vst1.f32    {d25[1]}, [%3]     \n"

								: "=r"(output0), // %0
								"=r"(output1), // %1
								"=r"(output2), // %2
								"=r"(output3), // %3
								"=r"(vb),      // %4
								"=r"(va)       // %5
								: "0"(output0),
								"1"(output1),
								"2"(output2),
								"3"(output3),
								"4"(vb),
								"5"(va),
								"r"(L),        // %12 
								"r"(biasptr)   // %13 
								: "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12"
								);
#endif // __aarch64__
#else
							float sum0 = biasptr[0];
							float sum1 = biasptr[1];
							float sum2 = biasptr[2];
							float sum3 = biasptr[3];

							for (int k = 0; k < L; k++)
							{
								sum0 += va[0] * vb[0];
								sum1 += va[1] * vb[0];
								sum2 += va[2] * vb[0];
								sum3 += va[3] * vb[0];

								va += 4;
								vb += 1;
							}

							output0[0] = sum0;
							output1[0] = sum1;
							output2[0] = sum2;
							output3[0] = sum3;
#endif // __ARM_NEON
							output0++;
							output1++;
							output2++;
							output3++;
						}
					}

					remain_outch_start += nn_outch << 2;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
					for (int i = remain_outch_start; i < outch; i++)
					{
						float* output = top_data + (i)* top_cstep;

						const float bias0 = this->bias_term_ ? bias[i] : 0.f;

						int j = 0;
						for (; j + 7 < N; j = j + 8)
						{
							const float* vb = bottom_tm_data + (j / 8) * bottom_tm_cstep;
#if __ARM_NEON && __aarch64__
							const float* va = kernel_tm_gemm_data + (i / 8 + (i % 8) / 4 + i % 4) * kernel_tm_gemm_cstep;
#else                
							const float* va = kernel_tm_gemm_data + (i / 4 + i % 4) * kernel_tm_gemm_cstep;
#endif // __ARM_NEON && __aarch64__

#if __ARM_NEON
#if __aarch64__
							asm volatile(
								"dup    v16.4s, %w7                  \n" // sum0
								"dup    v17.4s, %w7                  \n" // sum0n

								"lsr         w4, %w6, #2             \n"// r4 = nn = L >> 2
								"cmp         w4, #0                  \n"
								"beq         1f                      \n"

								"0:                                  \n"// for (; k+3<L; k=k+4)

								"prfm   pldl1keep, [%2, #128]        \n"
								"ld1    {v0.4s}, [%2], #16           \n"

								"prfm   pldl1keep, [%1, #128]                       \n"
								"ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%1], #64   \n" // data
								"ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%1], #64 \n"

								// k0
								"fmla    v16.4s, v8.4s, v0.s[0]      \n"// sum0 += (a00-a70) * k00
								"fmla    v17.4s, v9.4s, v0.s[0]      \n"//
																		// k1
								"fmla    v16.4s, v10.4s, v1.s[0]     \n"// sum0 += (a01-a71) * k01
								"fmla    v17.4s, v11.4s, v1.s[0]     \n"//
																		// k2
								"fmla    v16.4s, v12.4s, v2.s[0]     \n"// sum0 += (a02-a72) * k02
								"fmla    v17.4s, v13.4s, v2.s[0]     \n"//
																		// k3
								"fmla    v16.4s, v14.4s, v3.s[0]     \n"// sum0 += (a03-a73) * k03
								"fmla    v17.4s, v15.4s, v3.s[0]     \n"//

								"subs   w4, w4, #1                   \n"
								"bne    0b                           \n"

								"1:                                  \n"

								// remain loop
								"and    w4, %w6, #3                  \n"// w4 = remain = inch & 3;
								"cmp    w4, #0                       \n"
								"beq    3f                           \n"

								"2:                                  \n"
								"prfm   pldl1keep, [%2, #32]         \n"
								"ld1r   {v0.4s}, [%2], #4            \n"
								"prfm   pldl1keep, [%1, #256]        \n"
								"ld1    {v8.4s, v9.4s}, [%1], #32    \n"

								"subs   w4, w4, #1                   \n"
								// k0
								"fmla    v16.4s, v0.4s, v8.4s        \n"// sum0 += (a00-a70) * k00
								"fmla    v17.4s, v0.4s, v9.4s        \n"//

								"bne    2b                           \n"

								"3:                                  \n"
								"st1    {v16.4s, v17.4s}, [%0]       \n"

								: "=r"(output),  // %0
								"=r"(vb),      // %1
								"=r"(va)       // %2
								: "0"(output),
								"1"(vb),
								"2"(va),
								"r"(L),        // %6 
								"r"(bias0)     // %7
								: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17"
								);
#else
							asm volatile(
								"vdup.f32   q8, %7              \n"
								"vdup.f32   q9, %7              \n"
								// inch loop
								"lsr        r4, %6, #2          \n"// r4 = nn = inch >> 2
								"cmp        r4, #0              \n"
								"beq        1f                  \n"

								"0:                             \n"

								"pld        [%1, #512]          \n"
								"vldm       %1!, {d8-d15}       \n"
								"pld        [%2, #128]          \n"
								"vld1.f32   {d0-d1}, [%2]!      \n"

								"vmla.f32   q8, q4, d0[0]       \n"
								"vmla.f32   q9, q5, d0[0]       \n"

								"pld        [%1, #512]          \n"
								"vldm       %1!, {d24-d31}      \n"

								"vmla.f32   q8, q6, d0[1]       \n"
								"vmla.f32   q9, q7, d0[1]       \n"

								"subs       r4, r4, #1          \n"

								"vmla.f32   q8, q12, d1[0]      \n"
								"vmla.f32   q9, q13, d1[0]      \n"
								"vmla.f32   q8, q14, d1[1]      \n"
								"vmla.f32   q9, q15, d1[1]      \n"

								"bne        0b                  \n"

								"1:                             \n"
								// remain loop
								"and        r4, %6, #3          \n"// r4 = remain = inch & 3;
								"cmp        r4, #0              \n"
								"beq        3f                  \n"

								"2:                             \n"
								"pld        [%1, #256]          \n"
								"vld1.f32   {d8-d11}, [%1]!     \n"
								"pld        [%2, #32]           \n"
								"vld1.f32   {d0[],d1[]}, [%2]!  \n"

								"subs       r4, r4, #1          \n"

								"vmla.f32   q8, q4, q0          \n"
								"vmla.f32   q9, q5, q0          \n"
								"bne        2b                  \n"

								"3:                             \n"
								"vst1.f32   {d16-d19}, [%0]     \n"

								: "=r"(output), // %0
								"=r"(vb),     // %1
								"=r"(va)      // %2
								: "0"(output),
								"1"(vb),
								"2"(va),
								"r"(L),       // %6 
								"r"(bias0)    // %7 
								: "cc", "memory", "r4", "q0", "q4", "q5", "q6", "q7", "q8", "q9", "q12", "q13", "q14", "q15"
								);
#endif // __aarch64__
#else
							float sum[8] = { 0 };

							int k = 0;
							for (; k + 7 < L; k = k + 8)
							{
								for (int n = 0; n < 8; n++)
								{
									sum[n] += va[0] * vb[n];
									sum[n] += va[1] * vb[n + 8];
									sum[n] += va[2] * vb[n + 16];
									sum[n] += va[3] * vb[n + 24];
									sum[n] += va[4] * vb[n + 32];
									sum[n] += va[5] * vb[n + 40];
									sum[n] += va[6] * vb[n + 48];
									sum[n] += va[7] * vb[n + 56];
								}

								va += 8;
								vb += 64;
							}

							for (; k < L; k++)
							{
								for (int n = 0; n < 8; n++)
								{
									sum[n] += va[0] * vb[n];
								}

								va += 1;
								vb += 8;
							}

							for (int n = 0; n < 8; n++)
							{
								output[n] = sum[n] + bias0;
							}
#endif // __ARM_NEON
							output += 8;
						}

						for (; j < N; j++)
						{
							const float* vb = bottom_tm_data + (j / 8 + j % 8) * bottom_tm_cstep;
#if __ARM_NEON && __aarch64__
							const float* va = kernel_tm_gemm_data + (i / 8 + (i % 8) / 4 + i % 4) * kernel_tm_gemm_cstep;
#else                
							const float* va = kernel_tm_gemm_data + (i / 4 + i % 4) * kernel_tm_gemm_cstep;
#endif // __ARM_NEON && __aarch64__

							int k = 0;
#if __ARM_NEON
							float32x4_t _sum0 = vdupq_n_f32(0.f);

							for (; k + 3 < L; k += 4)
							{
								float32x4_t _p0 = vld1q_f32(vb);
								vb += 4;

								float32x4_t _k0 = vld1q_f32(va);
								va += 4;

#if __aarch64__
								_sum0 = vfmaq_f32(_sum0, _p0, _k0);
#else
								_sum0 = vmlaq_f32(_sum0, _p0, _k0);
#endif
							}

#if __aarch64__
							float sum0 = bias0 + vaddvq_f32(_sum0);
#else
							float32x2_t _ss = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
							float sum0 = bias0 + vget_lane_f32(vpadd_f32(_ss, _ss), 0);
#endif
#else
							float sum0 = bias0;
#endif // __ARM_NEON
							for (int k = 0; k < L; k++)
							{
								sum0 += va[0] * vb[0];

								va += 1;
								vb += 1;
							}
							output[0] = sum0;

							output++;
						}
					}
				}
			}
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
				else if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 2 && this->stride_w_ == 2))
				{
					conv3x3s2_packed_int8_neon(bottom_bordered, top_int32_);
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
		void operation_convolution_arm<Dtype>::conv3x3s1_winograd23_transform_kernel_int8_neon()
		{
			int inch = this->input_channel_;
			int outch = this->output_channel_;
			const int8_t* kernel = this->weights_i8_[0]->cpu_data();

			memory::tensor<short> kernel_tm(std::vector<int>{1, outch, inch, 4 * 4}, this->params_.device_, memory::NCHW);
			short* kernel_tm_data = kernel_tm.mutable_cpu_data();

			int kernel_tm_cstep = kernel_tm.count(2, 4);
			int kernel_tm_width = kernel_tm.width();

			// G
			const short ktm[4][3] = {
				{2, 0, 0},
				{1, 1, 1},
				{1, -1, 1},
				{0, 0, 2}
			};

			for (int p = 0; p < outch; p++)
			{
				for (int q = 0; q < inch; q++)
				{
					const signed char* kernel0 = (const signed char*)kernel + p * inch * 9 + q * 9;
					short* kernel_tm0 = kernel_tm_data + p * kernel_tm_cstep + (q) * kernel_tm_width;

					// transform kernel
					const signed char* k0 = kernel0;
					const signed char* k1 = kernel0 + 3;
					const signed char* k2 = kernel0 + 6;

					// h
					short tmp[4][3];
					for (int i = 0; i < 4; i++)
					{
						tmp[i][0] = (short)k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
						tmp[i][1] = (short)k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
						tmp[i][2] = (short)k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
					}

					// U
					for (int j = 0; j < 4; j++)
					{
						short* tmpp = &tmp[j][0];

						for (int i = 0; i < 4; i++)
						{
							kernel_tm0[j * 4 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
						}
					}
				}
			}

			for (int r = 0; r < 4; r++)
			{
				std::shared_ptr<memory::tensor<short>> kernel_tm_test(new memory::tensor<short>(std::vector<int>{1, outch / 8 + (outch % 8) / 4 + outch % 4, inch, 4 * 8}, this->params_.device_, memory::NCHW));
				short* kernel_tm_test_data = kernel_tm_test->mutable_cpu_data();
				int kernel_tm_test_cstep = kernel_tm_test->count(2, 4);
				int p = 0;
				for (; p + 7 < outch; p += 8)
				{
					const short* kernel0 = (const short*)kernel_tm_data + (p + 0) * inch * 16;
					const short* kernel1 = (const short*)kernel_tm_data + (p + 1) * inch * 16;
					const short* kernel2 = (const short*)kernel_tm_data + (p + 2) * inch * 16;
					const short* kernel3 = (const short*)kernel_tm_data + (p + 3) * inch * 16;
					const short* kernel4 = (const short*)kernel_tm_data + (p + 4) * inch * 16;
					const short* kernel5 = (const short*)kernel_tm_data + (p + 5) * inch * 16;
					const short* kernel6 = (const short*)kernel_tm_data + (p + 6) * inch * 16;
					const short* kernel7 = (const short*)kernel_tm_data + (p + 7) * inch * 16;

					short* ktmp = kernel_tm_test_data + (p / 8) * kernel_tm_test_cstep;

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
						kernel0 += 16;
						kernel1 += 16;
						kernel2 += 16;
						kernel3 += 16;
						kernel4 += 16;
						kernel5 += 16;
						kernel6 += 16;
						kernel7 += 16;
					}
				}

				for (; p + 3 < outch; p += 4)
				{
					const short* kernel0 = (const short*)kernel_tm_data + (p + 0) * inch * 16;
					const short* kernel1 = (const short*)kernel_tm_data + (p + 1) * inch * 16;
					const short* kernel2 = (const short*)kernel_tm_data + (p + 2) * inch * 16;
					const short* kernel3 = (const short*)kernel_tm_data + (p + 3) * inch * 16;

					short* ktmp = kernel_tm_test_data + (p / 8 + (p % 8) / 4) * kernel_tm_test_cstep;

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
						kernel0 += 16;
						kernel1 += 16;
						kernel2 += 16;
						kernel3 += 16;
					}
				}

				for (; p < outch; p++)
				{
					const short* kernel0 = (const short*)kernel_tm_data + p * inch * 16;

					short* ktmp = kernel_tm_test_data + (p / 8 + (p % 8) / 4 + p % 4) * kernel_tm_test_cstep;

					for (int q = 0; q < inch; q++)
					{
						ktmp[0] = kernel0[r * 4 + 0];
						ktmp[1] = kernel0[r * 4 + 1];
						ktmp[2] = kernel0[r * 4 + 2];
						ktmp[3] = kernel0[r * 4 + 3];

						ktmp += 4;
						kernel0 += 16;
					}
				}
				kernel_tm_int8_winograd_.push_back(kernel_tm_test);
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
				kernel_tm_int8_winograd_.push_back(kernel_tm_test);
			}
		}

		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv3x3s2_transform_kernel_int8_neon()
		{
			int inch = this->input_channel_;
			int outch = this->output_channel_;
			kernel_tm_int8_.reset(new memory::tensor<int8_t>(std::vector<int>{1, outch / 8 + outch % 8, inch, 8 * 9}, this->params_.device_, memory::NCHW));

			const signed char* kernel = this->weights_i8_[0]->cpu_data();
			int8_t* kernel_tm_int8_data = kernel_tm_int8_->mutable_cpu_data();
			int kernel_tm_int8_cstep = kernel_tm_int8_->count(2, 4);

			int p = 0;
			for (; p + 7 < outch; p += 8)
			{
				const signed char* k0 = kernel + (p + 0) * inch * 9;
				const signed char* k1 = kernel + (p + 1) * inch * 9;
				const signed char* k2 = kernel + (p + 2) * inch * 9;
				const signed char* k3 = kernel + (p + 3) * inch * 9;
				const signed char* k4 = kernel + (p + 4) * inch * 9;
				const signed char* k5 = kernel + (p + 5) * inch * 9;
				const signed char* k6 = kernel + (p + 6) * inch * 9;
				const signed char* k7 = kernel + (p + 7) * inch * 9;

				signed char* ktmp = kernel_tm_int8_data + (p / 8) * kernel_tm_int8_cstep;

				for (int q = 0; q < inch; q++)
				{
					for (int k = 0; k < 9; k++)
					{
						ktmp[0] = k0[k];
						ktmp[1] = k1[k];
						ktmp[2] = k2[k];
						ktmp[3] = k3[k];
						ktmp[4] = k4[k];
						ktmp[5] = k5[k];
						ktmp[6] = k6[k];
						ktmp[7] = k7[k];
						ktmp += 8;
					}

					k0 += 9;
					k1 += 9;
					k2 += 9;
					k3 += 9;
					k4 += 9;
					k5 += 9;
					k6 += 9;
					k7 += 9;
				}
			}
			for (; p < outch; p++)
			{
				const signed char* k0 = kernel + (p + 0) * inch * 9;

				signed char* ktmp = kernel_tm_int8_data + (p / 8 + p % 8) * kernel_tm_int8_cstep;

				for (int q = 0; q < inch; q++)
				{
					for (int k = 0; k < 9; k++)
					{
						ktmp[k] = k0[k];
					}
					ktmp += 9;

					k0 += 9;
				}
			}
		}

#if __aarch64__
		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv1x1s1_sgemm_transform_kernel_int8_neon()
		{
			int inch = this->input_channel_;
			int outch = this->output_channel_;
			kernel_tm_int8_.reset(new memory::tensor<int8_t>(std::vector<int>{1, 1, inch, outch}, this->params_.device_, memory::NCHW));
			int8_t* kernel_tm_int8_data = kernel_tm_int8_->mutable_cpu_data();
			const int8_t* kernel = this->weights_i8_[0]->cpu_data();
			reorder_a((int8_t*)kernel, kernel_tm_int8_data, outch, inch, inch);
		}
#else
		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv1x1s1_sgemm_transform_kernel_int8_neon()
		{
			int inch = this->input_channel_;
			int outch = this->output_channel_;

			kernel_tm_int8_.reset(new memory::tensor<int8_t>(std::vector<int>{1, outch / 4 + outch % 4, inch / 4 + inch % 4, 4 * 4}, this->params_.device_, memory::NCHW));
			int8_t* kernel_tm_int8_data = kernel_tm_int8_->mutable_cpu_data();
			int kernel_tm_int8_cstep = kernel_tm_int8_->count(2, 4);

			const int8_t* kernel = this->weights_i8_[0]->cpu_data();

			int p = 0;
			for (; p + 3 < outch; p += 4)
			{
				const signed char* kernel0 = kernel + (p + 0) * inch;
				const signed char* kernel1 = kernel + (p + 1) * inch;
				const signed char* kernel2 = kernel + (p + 2) * inch;
				const signed char* kernel3 = kernel + (p + 3) * inch;

				signed char* ktmp = kernel_tm_int8_data + (p / 4) * kernel_tm_int8_cstep;

				for (int q = 0; q < inch; q++)
				{
					// kernel0...3 0
					ktmp[0] = kernel0[0];
					ktmp[1] = kernel1[0];
					ktmp[2] = kernel2[0];
					ktmp[3] = kernel3[0];

					ktmp += 4;
					kernel0 += 1;
					kernel1 += 1;
					kernel2 += 1;
					kernel3 += 1;
				}
			}

			for (; p < outch; p++)
			{
				const signed char* kernel0 = kernel + p * inch;
				signed char* ktmp = kernel_tm_int8_data + (p / 4 + p % 4) * kernel_tm_int8_cstep;

				for (int q = 0; q < inch; q++)
				{
					ktmp[0] = kernel0[0];
					ktmp++;
					kernel0++;
				}
			}
		}
#endif

#if __aarch64__
		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv_im2col_sgemm_transform_kernel_int8_neon()
		{
			int inch = this->input_channel_;
			int outch = this->output_channel_;
			int kernel_size = this->kernel_size_h_ * this->kernel_size_w_;

			const int m = outch;
			const int k = inch * kernel_size;
			kernel_tm_int8_.reset(new memory::tensor<int8_t>(m * k, this->params_.device_, memory::NCHW));
			const int8_t* a = this->weights_i8_[0]->cpu_data();
			int8_t* sa = kernel_tm_int8_->mutable_cpu_data();
			reorder_a((int8_t*)a, sa, m, k, k);
		}
#else
		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv_im2col_sgemm_transform_kernel_int8_neon()
		{
			int inch = this->input_channel_;
			int outch = this->output_channel_;
			int kernel_size = this->kernel_size_h_ * this->kernel_size_w_;
			const signed char* kernel = this->weights_i8_[0]->cpu_data();

#if __ARM_NEON && __aarch64__
			// kernel memory packed 8 x 8
			kernel_tm_int8_.reset(new memory::tensor<int8_t>(std::vector<int>{1, outch / 8 + (outch % 8) / 4 + outch % 4, inch, 8 * kernel_size}, this->params_.device_, memory::NCHW));
#else
			// kernel memory packed 4 x 8
			kernel_tm_int8_.reset(new memory::tensor<int8_t>(std::vector<int>{1, outch / 4 + outch % 4, inch, 4 * kernel_size}, this->params_.device_, memory::NCHW));
#endif
			int8_t* kernel_tm_int8_data = kernel_tm_int8_->mutable_cpu_data();
			int kernel_tm_int8_cstep = kernel_tm_int8_->count(2, 4);

			int nn_outch = 0;
			int remain_outch_start = 0;

#if __ARM_NEON && __aarch64__
			nn_outch = outch >> 3;
			remain_outch_start = nn_outch << 3;

			for (int pp = 0; pp < nn_outch; pp++)
			{
				int p = pp * 8;

				const signed char* k0 = kernel + (p + 0) * inch * kernel_size;
				const signed char* k1 = kernel + (p + 1) * inch * kernel_size;
				const signed char* k2 = kernel + (p + 2) * inch * kernel_size;
				const signed char* k3 = kernel + (p + 3) * inch * kernel_size;
				const signed char* k4 = kernel + (p + 4) * inch * kernel_size;
				const signed char* k5 = kernel + (p + 5) * inch * kernel_size;
				const signed char* k6 = kernel + (p + 6) * inch * kernel_size;
				const signed char* k7 = kernel + (p + 7) * inch * kernel_size;

				signed char* ktmp = kernel_tm_int8_data + (p / 8) * kernel_tm_int8_cstep;

				for (int q = 0; q < inch * kernel_size; q++)
				{
					ktmp[0] = k0[0];
					ktmp[1] = k1[0];
					ktmp[2] = k2[0];
					ktmp[3] = k3[0];
					ktmp[4] = k4[0];
					ktmp[5] = k5[0];
					ktmp[6] = k6[0];
					ktmp[7] = k7[0];
					ktmp += 8;

					k0 += 1;
					k1 += 1;
					k2 += 1;
					k3 += 1;
					k4 += 1;
					k5 += 1;
					k6 += 1;
					k7 += 1;
				}
			}
#endif

			nn_outch = (outch - remain_outch_start) >> 2;

			for (int pp = 0; pp < nn_outch; pp++)
			{
				int p = remain_outch_start + pp * 4;

				const signed char* k0 = kernel + (p + 0) * inch * kernel_size;
				const signed char* k1 = kernel + (p + 1) * inch * kernel_size;
				const signed char* k2 = kernel + (p + 2) * inch * kernel_size;
				const signed char* k3 = kernel + (p + 3) * inch * kernel_size;

#if __ARM_NEON && __aarch64__
				signed char* ktmp = kernel_tm_int8_data + (p / 8 + (p % 8) / 4) * kernel_tm_int8_cstep;
#else
				signed char* ktmp = kernel_tm_int8_data + (p / 4) * kernel_tm_int8_cstep;
#endif // __ARM_NEON && __aarch64__

				for (int q = 0; q < inch * kernel_size; q++)
				{
					ktmp[0] = k0[0];
					ktmp[1] = k1[0];
					ktmp[2] = k2[0];
					ktmp[3] = k3[0];
					ktmp += 4;

					k0 += 1;
					k1 += 1;
					k2 += 1;
					k3 += 1;
				}
			}

			remain_outch_start += nn_outch << 2;

			for (int p = remain_outch_start; p < outch; p++)
			{
				const signed char* k0 = kernel + (p + 0) * inch * kernel_size;

#if __ARM_NEON && __aarch64__
				signed char* ktmp = kernel_tm_int8_data + (p / 8 + (p % 8) / 4 + p % 4) * kernel_tm_int8_cstep;
#else
				signed char* ktmp = kernel_tm_int8_data + (p / 4 + p % 4) * kernel_tm_int8_cstep;
#endif // __ARM_NEON && __aarch64__

				for (int q = 0; q < inch * kernel_size; q++)
				{
					ktmp[0] = k0[0];
					ktmp++;
					k0++;
				}
			}
		}
#endif

		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv3x3s1_winograd23_int8_neon(const std::shared_ptr<memory::tensor<int8_t>>& bottom, 
				std::shared_ptr<memory::tensor<int>>& top)
		{
			int num = bottom->num();
			int w = bottom->width();
			int h = bottom->height();
			int inch = bottom->channels();

			int outw = top->width();
			int outh = top->height();
			int outch = top->channels();

			// pad to 2n+2, winograd F(2,3)
			std::shared_ptr<memory::tensor<int8_t>> bottom_bordered = bottom;

			outw = (outw + 1) / 2 * 2;
			outh = (outh + 1) / 2 * 2;

			w = outw + 2;
			h = outh + 2;

			make_border(bottom, bottom_bordered, 0, h - bottom->height(), 0, w - bottom->width());
			int bottom_bordered_cstep = bottom_bordered->count(2, 4);

			// BEGIN transform input
			std::shared_ptr<memory::tensor<short>> bottom_tm;
			{
				int w_tm = outw / 2 * 4;
				int h_tm = outh / 2 * 4;

				int nColBlocks = h_tm / 4; // may be the block num in FeatherCNN
				int nRowBlocks = w_tm / 4;

				const int tiles = nColBlocks * nRowBlocks;

				bottom_tm = std::make_shared<memory::tensor<short>>(std::vector<int>{ num, tiles * 4, inch, 4}, this->params_.device_, memory::NCHW);
				int bottom_tm_cstep = bottom_tm->count(2, 4);

				for (int num_i = 0; num_i < num; num_i++)
				{
					const int8_t* bottom_bordered_data = bottom_bordered->cpu_data() + num_i * bottom_bordered->count(1, 4);
					short* bottom_tm_data = bottom_tm->mutable_cpu_data() + num_i * bottom_tm->count(1, 4);

					// BT
					// const float itm[4][4] = {
					//     {1.0f,  0.0f, -1.0f,  0.0f},
					//     {0.0f,  1.0f,  1.00f, 0.0f},
					//     {0.0f, -1.0f,  1.00f, 0.0f},
					//     {0.0f, -1.0f,  0.00f, 1.0f}
					// };

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int q = 0; q < inch; q++)
					{
						const signed char* img = bottom_bordered_data + (q) * bottom_bordered_cstep;

						for (int j = 0; j < nColBlocks; j++)
						{
							const signed char* r0 = img + w * j * 2;
							const signed char* r1 = r0 + w;
							const signed char* r2 = r1 + w;
							const signed char* r3 = r2 + w;

							for (int i = 0; i < nRowBlocks; i++)
							{
								short* out_tm0 = bottom_tm_data + (tiles * 0 + j * nRowBlocks + i) * bottom_tm_cstep + (q) * 4;
								short* out_tm1 = bottom_tm_data + (tiles * 1 + j * nRowBlocks + i) * bottom_tm_cstep + (q) * 4;
								short* out_tm2 = bottom_tm_data + (tiles * 2 + j * nRowBlocks + i) * bottom_tm_cstep + (q) * 4;
								short* out_tm3 = bottom_tm_data + (tiles * 3 + j * nRowBlocks + i) * bottom_tm_cstep + (q) * 4;
#if __ARM_NEON
#if __aarch64__
								asm volatile(
									// load
									"prfm   pldl1keep, [%0, #64]    \n"
									"ld1    {v0.8b}, [%0]           \n"
									"prfm   pldl1keep, [%1, #64]    \n"
									"ld1    {v1.8b}, [%1]           \n"
									"prfm   pldl1keep, [%2, #64]    \n"
									"ld1    {v2.8b}, [%2]           \n"
									"prfm   pldl1keep, [%3, #64]    \n"
									"ld1    {v3.8b}, [%3]           \n"
									// w = B_t * d, trans int8 to int16
									"ssubl    v4.8h, v0.8b, v2.8b   \n" // d4
									"saddl    v5.8h, v1.8b, v2.8b   \n" // d6
									"ssubl    v6.8h, v2.8b, v1.8b   \n" // d8
									"ssubl    v7.8h, v3.8b, v1.8b   \n" // d10
									// transpose w to w_t
									"trn1   v8.4h, v4.4h, v5.4h    \n"
									"trn2   v9.4h, v4.4h, v5.4h    \n"
									"trn1   v10.4h, v6.4h, v7.4h    \n"
									"trn2   v11.4h, v6.4h, v7.4h    \n"

									"trn1   v0.2s, v8.2s, v10.2s    \n"
									"trn2   v2.2s, v8.2s, v10.2s    \n"
									"trn1   v1.2s, v9.2s, v11.2s    \n"
									"trn2   v3.2s, v9.2s, v11.2s    \n"
									// U = B_t * d_t
									"sub    v4.4h, v0.4h, v2.4h   \n"
									"add    v5.4h, v1.4h, v2.4h   \n"
									"sub    v6.4h, v2.4h, v1.4h   \n"
									"sub    v7.4h, v3.4h, v1.4h   \n"
									// save
									"st1    {v4.4h}, [%4]   \n"
									"st1    {v5.4h}, [%5]   \n"
									"st1    {v6.4h}, [%6]   \n"
									"st1    {v7.4h}, [%7]   \n"
									: "=r"(r0),      // %0
									"=r"(r1),      // %1
									"=r"(r2),      // %2
									"=r"(r3),      // %3
									"=r"(out_tm0), // %4
									"=r"(out_tm1), // %5
									"=r"(out_tm2), // %6
									"=r"(out_tm3)  // %7
									: "0"(r0),
									"1"(r1),
									"2"(r2),
									"3"(r3),
									"4"(out_tm0),
									"5"(out_tm1),
									"6"(out_tm2),
									"7"(out_tm3)
									: "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
#else
								asm volatile(
									// load
									"pld         [%0, #64]     \n"
									"vld1.s8     {d0}, [%0]    \n"
									"pld         [%1, #64]     \n"
									"vld1.s8     {d1}, [%1]    \n"
									"pld         [%2, #64]     \n"
									"vld1.s8     {d2}, [%2]    \n"
									"pld         [%3, #64]     \n"
									"vld1.s8     {d3}, [%3]    \n"
									// w = B_t * d, trans int8 to int16
									"vsubl.s8    q2, d0, d2    \n" // d4
									"vaddl.s8    q3, d1, d2    \n" // d6
									"vsubl.s8    q4, d2, d1    \n" // d8
									"vsubl.s8    q5, d3, d1    \n" // d10
									// transpose w to w_t
									"vtrn.s16    d4, d6        \n"
									"vtrn.s16    d8, d10       \n"
									"vtrn.s32    d4, d8        \n"
									"vtrn.s32    d6, d10       \n"
									// U = B_t * d_t
									"vsub.s16    d11, d4, d8   \n"
									"vadd.s16    d12, d6, d8   \n"
									"vsub.s16    d13, d8, d6   \n"
									"vsub.s16    d14, d10, d6  \n"
									// save
									"vst1.s32    {d11}, [%4]   \n"
									"vst1.s32    {d12}, [%5]   \n"
									"vst1.s32    {d13}, [%6]   \n"
									"vst1.s32    {d14}, [%7]   \n"
									: "=r"(r0),      // %0
									"=r"(r1),      // %1
									"=r"(r2),      // %2
									"=r"(r3),      // %3
									"=r"(out_tm0), // %4
									"=r"(out_tm1), // %5
									"=r"(out_tm2), // %6
									"=r"(out_tm3)  // %7
									: "0"(r0),
									"1"(r1),
									"2"(r2),
									"3"(r3),
									"4"(out_tm0),
									"5"(out_tm1),
									"6"(out_tm2),
									"7"(out_tm3)
									: "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
#endif // __aarch64__
#else
								short d0[4], d1[4], d2[4], d3[4];
								short w0[4], w1[4], w2[4], w3[4];
								short t0[4], t1[4], t2[4], t3[4];
								// load
								for (int n = 0; n < 4; n++)
								{
									d0[n] = r0[n];
									d1[n] = r1[n];
									d2[n] = r2[n];
									d3[n] = r3[n];
								}
								// w = B_t * d
								for (int n = 0; n < 4; n++)
								{
									w0[n] = d0[n] - d2[n];
									w1[n] = d1[n] + d2[n];
									w2[n] = d2[n] - d1[n];
									w3[n] = d3[n] - d1[n];
								}
								// transpose d to d_t
								{
									t0[0] = w0[0];
									t1[0] = w0[1];
									t2[0] = w0[2];
									t3[0] = w0[3];
									t0[1] = w1[0];
									t1[1] = w1[1];
									t2[1] = w1[2];
									t3[1] = w1[3];
									t0[2] = w2[0];
									t1[2] = w2[1];
									t2[2] = w2[2];
									t3[2] = w2[3];
									t0[3] = w3[0];
									t1[3] = w3[1];
									t2[3] = w3[2];
									t3[3] = w3[3];
								}
								// U = B_t * d_t
								for (int n = 0; n < 4; n++)
								{
									d0[n] = t0[n] - t2[n];
									d1[n] = t1[n] + t2[n];
									d2[n] = t2[n] - t1[n];
									d3[n] = t3[n] - t1[n];
								}
								// save to out_tm
								for (int n = 0; n < 4; n++)
								{
									out_tm0[n] = d0[n];
									out_tm1[n] = d1[n];
									out_tm2[n] = d2[n];
									out_tm3[n] = d3[n];
								}
#endif
								r0 += 2;
								r1 += 2;
								r2 += 2;
								r3 += 2;
							}
						}
					}
				}
			}

			// BEGIN dot
			std::shared_ptr<memory::tensor<int>> top_tm;
			{
				int w_tm = outw / 2 * 4;
				int h_tm = outh / 2 * 4;

				int nColBlocks = h_tm / 4; // may be the block num in FeatherCNN
				int nRowBlocks = w_tm / 4;

				const int tiles = nColBlocks * nRowBlocks;

				top_tm = std::make_shared<memory::tensor<int>>(std::vector<int>{num, outch, tiles, 16}, this->params_.device_, memory::NCHW);
				int top_tm_cstep = top_tm->count(2, 4);
				int bottom_tm_cstep = bottom_tm->count(2, 4);

				for (size_t num_i = 0; num_i < num; num_i++)
				{
					const short* bottom_tm_data = bottom_tm->cpu_data() + num_i * bottom_tm->count(1, 4);
					int* top_tm_data = top_tm->mutable_cpu_data() + num_i * top_tm->count(1, 4);

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int r = 0; r < 4; r++)
					{
						const short* kernel_tm_int8_f23_data = kernel_tm_int8_winograd_[r]->cpu_data();
						int kernel_tm_int8_f23_cstep = kernel_tm_int8_winograd_[r]->count(2, 4);

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
								const short* kptr = kernel_tm_int8_f23_data + p / 8 * kernel_tm_int8_f23_cstep;
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
									"prfm    pldl1keep, [%9, #128]    \n" // _r0 = vld1_s16(r0);  // input inch0
									"ld1     {v8.4h}, [%8]            \n"
									"ld1     {v9.4h, v10.4h}, [%9]    \n" // _k0 = vld1q_s16(kptr);
									"add     %9, %9, #16              \n"
									"ld1     {v11.4h, v12.4h}, [%9]   \n" // _k0n = vld1q_s16(kptr+8);
									"add     %9, %9, #16              \n"
									"ld1     {v13.4h, v14.4h}, [%9]   \n" // _k1 = vld1q_s16(kptr+16);
									"add     %9, %9, #16              \n"
									"ld1     {v15.4h, v16.4h}, [%9]   \n" // _k1n = vld1q_s16(kptr+24);
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
									"vld1.s16    {d18-d19}, [%9]  \n" // _k0 = vld1q_s16(kptr);
									"add         %9, #16          \n"
									"vld1.s16    {d20-d21}, [%9]  \n" // _k0n = vld1q_s16(kptr+8);
									"add         %9, #16          \n"
									"vld1.s16    {d22-d23}, [%9]  \n" // _k1 = vld1q_s16(kptr+16);
									"add         %9, #16          \n"
									"vld1.s16    {d24-d25}, [%9]  \n" // _k1n = vld1q_s16(kptr+24);
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
								output0_tm += 16;
								output1_tm += 16;
								output2_tm += 16;
								output3_tm += 16;
								output4_tm += 16;
								output5_tm += 16;
								output6_tm += 16;
								output7_tm += 16;
							}
						}

						nn_outch = (outch - remain_outch_start) >> 2;

						for (int pp = 0; pp < nn_outch; pp++)
						{
							int p = remain_outch_start + pp * 4;

							int* output0_tm = top_tm_data + top_tm_cstep * (p);
							int* output1_tm = top_tm_data + top_tm_cstep * (p + 1);
							int* output2_tm = top_tm_data + top_tm_cstep * (p + 2);
							int* output3_tm = top_tm_data + top_tm_cstep * (p + 3);

							output0_tm = output0_tm + r * 4;
							output1_tm = output1_tm + r * 4;
							output2_tm = output2_tm + r * 4;
							output3_tm = output3_tm + r * 4;

							for (int i = 0; i < tiles; i++)
							{
								const short* kptr = kernel_tm_int8_f23_data + (p / 8 + (p % 8) / 4) * kernel_tm_int8_f23_cstep;
								const short* r0 = bottom_tm_data + bottom_tm_cstep * (tiles * r + i);
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
									"ld1     {v9.4h, v10.4h}, [%5]    \n" // _k0 = vld1q_s16(kptr);
									"add     %5, %5, #16              \n"
									"ld1     {v11.4h, v12.4h}, [%5]   \n" // _k0n = vld1q_s16(kptr+8);
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
									"vld1.s16    {d18-d19}, [%5]  \n" // _k0 = vld1q_s16(kptr);
									"add         %5, #16          \n"
									"vld1.s16    {d20-d21}, [%5]  \n" // _k0n = vld1q_s16(kptr+8);
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
								output0_tm += 16;
								output1_tm += 16;
								output2_tm += 16;
								output3_tm += 16;
							}
						}

						remain_outch_start += nn_outch << 2;

						for (int p = remain_outch_start; p < outch; p++)
						{
							int* output0_tm = top_tm_data + (p) * top_tm_cstep;

							output0_tm = output0_tm + r * 4;

							for (int i = 0; i < tiles; i++)
							{
								const short* kptr = kernel_tm_int8_f23_data + (p / 8 + (p % 8) / 4 + p % 4) * kernel_tm_int8_f23_cstep;
								const short* r0 = bottom_tm_data + bottom_tm_cstep * (tiles * r + i);

#if __ARM_NEON
#if __aarch64__
								asm volatile(
									// inch loop
									"eor    v0.16b, v0.16b, v0.16b    \n"
									"mov    w4, %w6                   \n"

									"0:                               \n" // for (int q=0; q<inch; q++)
									//"prfm    pldl1keep, [%2, #128]    \n" // _r0 = vld1_s16(r0);  // input inch0
									"ld1     {v8.4h}, [%1]            \n"
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
#else
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
#endif
								output0_tm += 16;
							}
						}
					}
				}
			}
			// END dot

			// BEGIN transform output
			std::shared_ptr<memory::tensor<int>> top_bordered(new memory::tensor<int>(std::vector<int>{ num, outch, outh, outw }, this->params_.device_, memory::NCHW));
			{
				// AT
				// const float itm[2][4] = {
				//     {1.0f,  1.0f,  1.0f,  0.0f},
				//     {0.0f,  1.0f, -1.0f,  1.0f}
				// };

				int w_tm = outw / 2 * 4;
				int h_tm = outh / 2 * 4;

				int nColBlocks = h_tm / 4; // may be the block num in FeatherCNN
				int nRowBlocks = w_tm / 4;

				int top_tm_cstep = top_tm->count(2, 4);
				int top_bordered_cstep = top_bordered->count(2, 4);
				for (size_t num_i = 0; num_i < num; num_i++)
				{
					const int* top_tm_data = top_tm->cpu_data() + num_i * top_tm->count(1, 4);
					int* top_bordered_data = top_bordered->mutable_cpu_data() + num_i * top_bordered->count(1, 4);
#if __ARM_NEON
					int32x2_t _shift = vdup_n_s32(-2);
#endif

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int p = 0; p < outch; p++)
					{
						const int* out_tile = top_tm_data + (p) * top_tm_cstep;
						int* outRow0 = top_bordered_data + (p) * top_bordered_cstep;
						int* outRow1 = outRow0 + outw;

						for (int j = 0; j < nColBlocks; j++)
						{
							for (int i = 0; i < nRowBlocks; i++)
							{
#if __ARM_NEON
#if __aarch64__
								asm volatile(
									"prfm   pldl1keep, [%0, #512]  \n"
									"ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64    \n"

									"add    v0.4s, v0.4s, v1.4s    \n" // s0 = s0 + s1 + s2;
									"sub    v1.4s, v1.4s, v2.4s    \n"
									"add    v0.4s, v0.4s, v2.4s    \n" // s1 = s1 - s2 + s3;
									"add    v1.4s, v1.4s, v3.4s    \n"

									"trn1   v4.4s, v0.4s, v1.4s    \n"
									"trn2   v5.4s, v0.4s, v1.4s    \n"

									"dup    v6.2d, v4.d[1]         \n"
									"dup    v7.2d, v5.d[1]         \n"

									"add    v0.2s, v4.2s, v5.2s    \n" // o0 = d0 + d1 + d2;
									"sub    v1.2s, v5.2s, v6.2s    \n"
									"add    v0.2s, v0.2s, v6.2s    \n" // o1 = d1 - d2 + d3;
									"add    v1.2s, v1.2s, v7.2s    \n"

									"sshl    v0.2s, v0.2s, %6.2s   \n" // o0 = o0 >> 2
									"sshl    v1.2s, v1.2s, %6.2s   \n" // o1 = o1 >> 2

									"st1     {v0.2s}, [%1], #8     \n"
									"st1     {v1.2s}, [%2], #8     \n"
									: "=r"(out_tile), // %0
									"=r"(outRow0),  // %1
									"=r"(outRow1)   // %2
									: "0"(out_tile),
									"1"(outRow0),
									"2"(outRow1),
									"w"(_shift) // %6
									: "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else
								asm volatile(
									"pld        [%0, #512]      \n"
									"vldm        %0!, {d0-d7}   \n"

									"vaddq.s32    q0, q0, q1    \n" // s0 = s0 + s1 + s2;
									"vsubq.s32    q1, q1, q2    \n"
									"vaddq.s32    q0, q0, q2    \n" // s1 = s1 - s2 + s3;
									"vaddq.s32    q1, q1, q3    \n"

									"vtrn.s32    q0, q1         \n"

									"vadd.s32    d8, d0, d2     \n" // o0 = d0 + d1 + d2;
									"vsub.s32    d9, d2, d1     \n"
									"vadd.s32    d8, d8, d1     \n" // o1 = d1 - d2 + d3;
									"vadd.s32    d9, d9, d3     \n"

									"vshl.s32    d8, d8, %P6    \n" // o0 = o0 >> 2
									"vshl.s32    d9, d9, %P6    \n" // o1 = o1 >> 2

									"vst1.s32    {d8}, [%1]!    \n"
									"vst1.s32    {d9}, [%2]!    \n"
									: "=r"(out_tile), // %0
									"=r"(outRow0),  // %1
									"=r"(outRow1)   // %2
									: "0"(out_tile),
									"1"(outRow0),
									"2"(outRow1),
									"w"(_shift) // %6
									: "cc", "memory", "q0", "q1", "q2", "q3", "q4");
#endif // __aarch64__
#else
								int s0[4], s1[4], s2[4], s3[4];
								int w0[4], w1[4];
								int d0[2], d1[2], d2[2], d3[2];
								int o0[2], o1[2];
								// load
								for (int n = 0; n < 4; n++)
								{
									s0[n] = out_tile[n];
									s1[n] = out_tile[n + 4];
									s2[n] = out_tile[n + 8];
									s3[n] = out_tile[n + 12];
								}
								// w = A_T * W
								for (int n = 0; n < 4; n++)
								{
									w0[n] = s0[n] + s1[n] + s2[n];
									w1[n] = s1[n] - s2[n] + s3[n];
								}
								// transpose w to w_t
								{
									d0[0] = w0[0];
									d0[1] = w1[0];
									d1[0] = w0[1];
									d1[1] = w1[1];
									d2[0] = w0[2];
									d2[1] = w1[2];
									d3[0] = w0[3];
									d3[1] = w1[3];
								}
								// Y = A_T * w_t
								for (int n = 0; n < 2; n++)
								{
									o0[n] = d0[n] + d1[n] + d2[n];
									o1[n] = d1[n] - d2[n] + d3[n];
								}
								// save to top blob tm,why right 2,because the G' = G*2
								outRow0[0] = o0[0] >> 2;
								outRow0[1] = o0[1] >> 2;
								outRow1[0] = o1[0] >> 2;
								outRow1[1] = o1[1] >> 2;

								out_tile += 16;

								outRow0 += 2;
								outRow1 += 2;
#endif // __ARM_NEON
							}

							outRow0 += outw;
							outRow1 += outw;
						}
					}
				}
			}
			// END transform output

			// cut result pad
			cut_border_cpu(top_bordered, top, 0, top_bordered->height() - top->height(), 0, top_bordered->width() - top->width());
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
						const short* kernel_tm_int8_f43_data = kernel_tm_int8_winograd_[r]->cpu_data();
						int kernel_tm_int8_f43_cstep = kernel_tm_int8_winograd_[r]->count(2, 4);

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
								const short* kptr = kernel_tm_int8_f43_data + p / 8 * kernel_tm_int8_f43_cstep;
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
								const short* kptr = kernel_tm_int8_f43_data + (p / 8 + (p % 8) / 4) * kernel_tm_int8_f43_cstep;
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
								const short* kptr = kernel_tm_int8_f43_data + (p / 8 + (p % 8) / 4 + p % 4) * kernel_tm_int8_f43_cstep;
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
		void operation_convolution_arm<Dtype>::conv3x3s2_packed_int8_neon(const std::shared_ptr<memory::tensor<int8_t>>& bottom, std::shared_ptr<memory::tensor<int>>& top)
		{
			int num = bottom->num();
			int w = bottom->width();
			int h = bottom->height();
			int inch = bottom->channels();
			int bottom_cstep = w * h;

			int outw = top->width();
			int outh = top->height();
			int outch = top->channels();
			int top_cstep = outw * outh;

			const int tailstep = w - 2 * outw + w;

			const int8_t* kernel_tm_int8_data = kernel_tm_int8_->cpu_data();
			int kernel_tm_int8_cstep = kernel_tm_int8_->count(2, 4);

			for (size_t num_i = 0; num_i < num; num_i++)
			{
				const int8_t* bottom_data = bottom->cpu_data() + num_i * bottom->count(1, 4);
				int* top_data = top->mutable_cpu_data() + num_i * top->count(1, 4);

				int nn_outch = outch >> 3;
				int remain_outch_start = nn_outch << 3;
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
				for (int pp = 0; pp < nn_outch; pp++)
				{
					int p = pp * 8;

					int* out0 = top_data + (p + 0) * top_cstep;
					int* out1 = top_data + (p + 1) * top_cstep;
					int* out2 = top_data + (p + 2) * top_cstep;
					int* out3 = top_data + (p + 3) * top_cstep;
					int* out4 = top_data + (p + 4) * top_cstep;
					int* out5 = top_data + (p + 5) * top_cstep;
					int* out6 = top_data + (p + 6) * top_cstep;
					int* out7 = top_data + (p + 7) * top_cstep;

					fill(out0, top_cstep, 0);
					fill(out1, top_cstep, 0);
					fill(out2, top_cstep, 0);
					fill(out3, top_cstep, 0);
					fill(out4, top_cstep, 0);
					fill(out5, top_cstep, 0);
					fill(out6, top_cstep, 0);
					fill(out7, top_cstep, 0);

					const signed char* ktmp = kernel_tm_int8_data + (p / 8) * kernel_tm_int8_cstep;

					for (int q = 0; q < inch; q++)
					{
						int* outptr0 = out0;
						int* outptr1 = out1;
						int* outptr2 = out2;
						int* outptr3 = out3;
						int* outptr4 = out4;
						int* outptr5 = out5;
						int* outptr6 = out6;
						int* outptr7 = out7;

						const signed char* img0 = bottom_data + q * bottom_cstep;

						const signed char* r0 = img0;
						const signed char* r1 = img0 + w;
						const signed char* r2 = img0 + w * 2;

						int i = 0;

						for (; i < outh; i++)
						{
#if __ARM_NEON
#if __aarch64__
							int nn = outw >> 3;
							int remain = outw & 7;
#else
							int nn = outw >> 2;
							int remain = outw & 3;
#endif // __aarch64__
#else
							int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
							if (nn > 0)
							{
								asm volatile(
									"0:                                   \n"

									"ld1    {v0.8b, v1.8b, v2.8b}, [%12], #24  \n" //ktmp
									"ld2    {v3.8b, v4.8b}, [%9], #16     \n"      //r0-r2
									"ld2    {v5.8b, v6.8b}, [%9]          \n"

									"ld1    {v8.4s, v9.4s}, [%1]          \n" //out0
									"ld1    {v10.4s, v11.4s}, [%2]        \n" //out1
									"ld1    {v12.4s, v13.4s}, [%3]        \n" //out2
									"ld1    {v14.4s, v15.4s}, [%4]        \n" //out3
									"ld1    {v16.4s, v17.4s}, [%5]        \n" //out4
									"ld1    {v18.4s, v19.4s}, [%6]        \n" //out5
									"ld1    {v20.4s, v21.4s}, [%7]        \n" //out6
									"ld1    {v22.4s, v23.4s}, [%8]        \n" //out7

									"ext    v7.8b, v3.8b, v5.8b, #1       \n"

									"sshll  v0.8h, v0.8b, #0              \n" //(k00-k70)
									"sshll  v1.8h, v1.8b, #0              \n" //(k01-k71)
									"sshll  v2.8h, v2.8b, #0              \n" //(k02-k72)
									"sshll  v3.8h, v3.8b, #0              \n" // r0
									"sshll  v4.8h, v4.8b, #0              \n" // r1
									"sshll  v7.8h, v7.8b, #0              \n" // r2

									// r0
									"smlal  v8.4s, v3.4h, v0.h[0]         \n" // out0 += (r00-r07)*k00
									"smlal2  v9.4s, v3.8h, v0.h[0]        \n"
									"smlal  v10.4s, v3.4h, v0.h[1]        \n" // out1 += (r00-r07)*k10
									"smlal2  v11.4s, v3.8h, v0.h[1]       \n"
									"smlal  v12.4s, v3.4h, v0.h[2]        \n" // out2 += (r00-r07)*k20
									"smlal2  v13.4s, v3.8h, v0.h[2]       \n"
									"smlal  v14.4s, v3.4h, v0.h[3]        \n" // out3 += (r00-r07)*k30
									"smlal2  v15.4s, v3.8h, v0.h[3]       \n"
									"smlal  v16.4s, v3.4h, v0.h[4]        \n" // out4 += (r00-r07)*k40
									"smlal2  v17.4s, v3.8h, v0.h[4]       \n"
									"smlal  v18.4s, v3.4h, v0.h[5]        \n" // out5 += (r00-r07)*k50
									"smlal2  v19.4s, v3.8h, v0.h[5]       \n"
									"smlal  v20.4s, v3.4h, v0.h[6]        \n" // out6 += (r00-r07)*k60
									"smlal2  v21.4s, v3.8h, v0.h[6]       \n"
									"smlal  v22.4s, v3.4h, v0.h[7]        \n" // out7 += (r00-r07)*k70
									"smlal2  v23.4s, v3.8h, v0.h[7]       \n"
									// r1
									"smlal  v8.4s, v4.4h, v1.h[0]         \n" // out0 += (r10-r17)*k01
									"smlal2  v9.4s, v4.8h, v1.h[0]        \n"
									"smlal  v10.4s, v4.4h, v1.h[1]        \n" // out1 += (r10-r17)*k11
									"smlal2  v11.4s, v4.8h, v1.h[1]       \n"
									"smlal  v12.4s, v4.4h, v1.h[2]        \n" // out2 += (r10-r17)*k21
									"smlal2  v13.4s, v4.8h, v1.h[2]       \n"
									"smlal  v14.4s, v4.4h, v1.h[3]        \n" // out3 += (r10-r17)*k31
									"smlal2  v15.4s, v4.8h, v1.h[3]       \n"
									"smlal  v16.4s, v4.4h, v1.h[4]        \n" // out4 += (r10-r17)*k41
									"smlal2  v17.4s, v4.8h, v1.h[4]       \n"
									"smlal  v18.4s, v4.4h, v1.h[5]        \n" // out5 += (r10-r17)*k51
									"smlal2  v19.4s, v4.8h, v1.h[5]       \n"
									"smlal  v20.4s, v4.4h, v1.h[6]        \n" // out6 += (r10-r17)*k61
									"smlal2  v21.4s, v4.8h, v1.h[6]       \n"
									"smlal  v22.4s, v4.4h, v1.h[7]        \n" // out7 += (r10-r17)*k71
									"smlal2  v23.4s, v4.8h, v1.h[7]       \n"
									// r2
									"smlal  v8.4s, v7.4h, v2.h[0]         \n" // out0 += (r20-r27)*k02
									"smlal2  v9.4s, v7.8h, v2.h[0]        \n"
									"smlal  v10.4s, v7.4h, v2.h[1]        \n" // out1 += (r20-r27)*k12
									"smlal2  v11.4s, v7.8h, v2.h[1]       \n"
									"smlal  v12.4s, v7.4h, v2.h[2]        \n" // out2 += (r20-r27)*k22
									"smlal2  v13.4s, v7.8h, v2.h[2]       \n"
									"smlal  v14.4s, v7.4h, v2.h[3]        \n" // out3 += (r20-r27)*k32
									"smlal2  v15.4s, v7.8h, v2.h[3]       \n"
									"smlal  v16.4s, v7.4h, v2.h[4]        \n" // out4 += (r20-r27)*k42
									"smlal2  v17.4s, v7.8h, v2.h[4]       \n"
									"smlal  v18.4s, v7.4h, v2.h[5]        \n" // out5 += (r20-r27)*k52
									"smlal2  v19.4s, v7.8h, v2.h[5]       \n"
									"smlal  v20.4s, v7.4h, v2.h[6]        \n" // out6 += (r20-r27)*k62
									"smlal2  v21.4s, v7.8h, v2.h[6]       \n"
									"smlal  v22.4s, v7.4h, v2.h[7]        \n" // out7 += (r20-r27)*k72
									"smlal2  v23.4s, v7.8h, v2.h[7]       \n"

									"ld1    {v0.8b, v1.8b, v2.8b}, [%12], #24  \n" //ktmp
									"ld2    {v3.8b, v4.8b}, [%10], #16    \n"      //r3-r5
									"ld2    {v5.8b, v6.8b}, [%10]         \n"

									"ext    v7.8b, v3.8b, v5.8b, #1       \n"

									"sshll  v0.8h, v0.8b, #0              \n" //(k03-k73)
									"sshll  v1.8h, v1.8b, #0              \n" //(k04-k74)
									"sshll  v2.8h, v2.8b, #0              \n" //(k05-k75)
									"sshll  v3.8h, v3.8b, #0              \n" // r3
									"sshll  v4.8h, v4.8b, #0              \n" // r4
									"sshll  v7.8h, v7.8b, #0              \n" // r5

									// r3
									"smlal  v8.4s, v3.4h, v0.h[0]         \n" // out0 += (r30-r37)*k03
									"smlal2  v9.4s, v3.8h, v0.h[0]        \n"
									"smlal  v10.4s, v3.4h, v0.h[1]        \n" // out1 += (r30-r37)*k13
									"smlal2  v11.4s, v3.8h, v0.h[1]       \n"
									"smlal  v12.4s, v3.4h, v0.h[2]        \n" // out2 += (r30-r37)*k23
									"smlal2  v13.4s, v3.8h, v0.h[2]       \n"
									"smlal  v14.4s, v3.4h, v0.h[3]        \n" // out3 += (r30-r37)*k33
									"smlal2  v15.4s, v3.8h, v0.h[3]       \n"
									"smlal  v16.4s, v3.4h, v0.h[4]        \n" // out4 += (r30-r37)*k43
									"smlal2  v17.4s, v3.8h, v0.h[4]       \n"
									"smlal  v18.4s, v3.4h, v0.h[5]        \n" // out5 += (r30-r37)*k53
									"smlal2  v19.4s, v3.8h, v0.h[5]       \n"
									"smlal  v20.4s, v3.4h, v0.h[6]        \n" // out6 += (r30-r37)*k63
									"smlal2  v21.4s, v3.8h, v0.h[6]       \n"
									"smlal  v22.4s, v3.4h, v0.h[7]        \n" // out7 += (r30-r37)*k73
									"smlal2  v23.4s, v3.8h, v0.h[7]       \n"
									// r4
									"smlal  v8.4s, v4.4h, v1.h[0]         \n" // out0 += (r40-r47)*k04
									"smlal2  v9.4s, v4.8h, v1.h[0]        \n"
									"smlal  v10.4s, v4.4h, v1.h[1]        \n" // out1 += (r40-r47)*k14
									"smlal2  v11.4s, v4.8h, v1.h[1]       \n"
									"smlal  v12.4s, v4.4h, v1.h[2]        \n" // out2 += (r40-r47)*k24
									"smlal2  v13.4s, v4.8h, v1.h[2]       \n"
									"smlal  v14.4s, v4.4h, v1.h[3]        \n" // out3 += (r40-r47)*k34
									"smlal2  v15.4s, v4.8h, v1.h[3]       \n"
									"smlal  v16.4s, v4.4h, v1.h[4]        \n" // out4 += (r40-r47)*k44
									"smlal2  v17.4s, v4.8h, v1.h[4]       \n"
									"smlal  v18.4s, v4.4h, v1.h[5]        \n" // out5 += (r40-r47)*k54
									"smlal2  v19.4s, v4.8h, v1.h[5]       \n"
									"smlal  v20.4s, v4.4h, v1.h[6]        \n" // out6 += (r40-r47)*k64
									"smlal2  v21.4s, v4.8h, v1.h[6]       \n"
									"smlal  v22.4s, v4.4h, v1.h[7]        \n" // out7 += (r40-r47)*k74
									"smlal2  v23.4s, v4.8h, v1.h[7]       \n"
									// r5
									"smlal  v8.4s, v7.4h, v2.h[0]         \n" // out0 += (r50-r57)*k05
									"smlal2  v9.4s, v7.8h, v2.h[0]        \n"
									"smlal  v10.4s, v7.4h, v2.h[1]        \n" // out1 += (r50-r57)*k15
									"smlal2  v11.4s, v7.8h, v2.h[1]       \n"
									"smlal  v12.4s, v7.4h, v2.h[2]        \n" // out2 += (r50-r57)*k25
									"smlal2  v13.4s, v7.8h, v2.h[2]       \n"
									"smlal  v14.4s, v7.4h, v2.h[3]        \n" // out3 += (r50-r57)*k35
									"smlal2  v15.4s, v7.8h, v2.h[3]       \n"
									"smlal  v16.4s, v7.4h, v2.h[4]        \n" // out4 += (r50-r57)*k45
									"smlal2  v17.4s, v7.8h, v2.h[4]       \n"
									"smlal  v18.4s, v7.4h, v2.h[5]        \n" // out5 += (r50-r57)*k55
									"smlal2  v19.4s, v7.8h, v2.h[5]       \n"
									"smlal  v20.4s, v7.4h, v2.h[6]        \n" // out6 += (r50-r57)*k65
									"smlal2  v21.4s, v7.8h, v2.h[6]       \n"
									"smlal  v22.4s, v7.4h, v2.h[7]        \n" // out7 += (r50-r57)*k75
									"smlal2  v23.4s, v7.8h, v2.h[7]       \n"

									"ld1    {v0.8b, v1.8b, v2.8b}, [%12], #24  \n" //ktmp
									"ld2    {v3.8b, v4.8b}, [%11], #16    \n"      //r6-r8
									"ld2    {v5.8b, v6.8b}, [%11]         \n"

									"ext    v7.8b, v3.8b, v5.8b, #1       \n"

									"sshll  v0.8h, v0.8b, #0              \n" //(k06-k76)
									"sshll  v1.8h, v1.8b, #0              \n" //(k07-k77)
									"sshll  v2.8h, v2.8b, #0              \n" //(k08-k78)
									"sshll  v3.8h, v3.8b, #0              \n" // r6
									"sshll  v4.8h, v4.8b, #0              \n" // r7
									"sshll  v7.8h, v7.8b, #0              \n" // r8

									// r6
									"smlal  v8.4s, v3.4h, v0.h[0]         \n" // out0 += (r60-r67)*k06
									"smlal2  v9.4s, v3.8h, v0.h[0]        \n"
									"smlal  v10.4s, v3.4h, v0.h[1]        \n" // out1 += (r60-r67)*k16
									"smlal2  v11.4s, v3.8h, v0.h[1]       \n"
									"smlal  v12.4s, v3.4h, v0.h[2]        \n" // out2 += (r60-r67)*k26
									"smlal2  v13.4s, v3.8h, v0.h[2]       \n"
									"smlal  v14.4s, v3.4h, v0.h[3]        \n" // out3 += (r60-r67)*k36
									"smlal2  v15.4s, v3.8h, v0.h[3]       \n"
									"smlal  v16.4s, v3.4h, v0.h[4]        \n" // out4 += (r60-r67)*k46
									"smlal2  v17.4s, v3.8h, v0.h[4]       \n"
									"smlal  v18.4s, v3.4h, v0.h[5]        \n" // out5 += (r60-r67)*k56
									"smlal2  v19.4s, v3.8h, v0.h[5]       \n"
									"smlal  v20.4s, v3.4h, v0.h[6]        \n" // out6 += (r60-r67)*k66
									"smlal2  v21.4s, v3.8h, v0.h[6]       \n"
									"smlal  v22.4s, v3.4h, v0.h[7]        \n" // out7 += (r60-r67)*k76
									"smlal2  v23.4s, v3.8h, v0.h[7]       \n"
									// r7
									"smlal  v8.4s, v4.4h, v1.h[0]         \n" // out0 += (r70-r77)*k07
									"smlal2  v9.4s, v4.8h, v1.h[0]        \n"
									"smlal  v10.4s, v4.4h, v1.h[1]        \n" // out1 += (r70-r77)*k17
									"smlal2  v11.4s, v4.8h, v1.h[1]       \n"
									"smlal  v12.4s, v4.4h, v1.h[2]        \n" // out2 += (r70-r77)*k27
									"smlal2  v13.4s, v4.8h, v1.h[2]       \n"
									"smlal  v14.4s, v4.4h, v1.h[3]        \n" // out3 += (r70-r77)*k37
									"smlal2  v15.4s, v4.8h, v1.h[3]       \n"
									"smlal  v16.4s, v4.4h, v1.h[4]        \n" // out4 += (r70-r77)*k47
									"smlal2  v17.4s, v4.8h, v1.h[4]       \n"
									"smlal  v18.4s, v4.4h, v1.h[5]        \n" // out5 += (r70-r77)*k57
									"smlal2  v19.4s, v4.8h, v1.h[5]       \n"
									"smlal  v20.4s, v4.4h, v1.h[6]        \n" // out6 += (r70-r77)*k67
									"smlal2  v21.4s, v4.8h, v1.h[6]       \n"
									"smlal  v22.4s, v4.4h, v1.h[7]        \n" // out7 += (r70-r77)*k77
									"smlal2  v23.4s, v4.8h, v1.h[7]       \n"
									// r8
									"smlal  v8.4s, v7.4h, v2.h[0]         \n" // out0 += (r80-r87)*k08
									"smlal2  v9.4s, v7.8h, v2.h[0]        \n"
									"smlal  v10.4s, v7.4h, v2.h[1]        \n" // out1 += (r80-r87)*k18
									"smlal2  v11.4s, v7.8h, v2.h[1]       \n"
									"smlal  v12.4s, v7.4h, v2.h[2]        \n" // out2 += (r80-r87)*k28
									"smlal2  v13.4s, v7.8h, v2.h[2]       \n"
									"smlal  v14.4s, v7.4h, v2.h[3]        \n" // out3 += (r80-r87)*k38
									"smlal2  v15.4s, v7.8h, v2.h[3]       \n"
									"smlal  v16.4s, v7.4h, v2.h[4]        \n" // out4 += (r80-r87)*k48
									"smlal2  v17.4s, v7.8h, v2.h[4]       \n"
									"smlal  v18.4s, v7.4h, v2.h[5]        \n" // out5 += (r80-r87)*k58
									"smlal2  v19.4s, v7.8h, v2.h[5]       \n"
									"smlal  v20.4s, v7.4h, v2.h[6]        \n" // out6 += (r80-r87)*k68
									"smlal2  v21.4s, v7.8h, v2.h[6]       \n"
									"smlal  v22.4s, v7.4h, v2.h[7]        \n" // out7 += (r80-r87)*k78
									"smlal2  v23.4s, v7.8h, v2.h[7]       \n"

									"st1    {v8.4s, v9.4s}, [%1], #32     \n"
									"st1    {v10.4s, v11.4s}, [%2], #32   \n"
									"st1    {v12.4s, v13.4s}, [%3], #32   \n"
									"st1    {v14.4s, v15.4s}, [%4], #32   \n"
									"st1    {v16.4s, v17.4s}, [%5], #32   \n"
									"st1    {v18.4s, v19.4s}, [%6], #32   \n"
									"st1    {v20.4s, v21.4s}, [%7], #32   \n"
									"st1    {v22.4s, v23.4s}, [%8], #32   \n"

									"subs   %w0, %w0, #1                  \n"
									"sub    %12, %12, #72                 \n" // reset ktmp

									"bne    0b                            \n"

									: "=r"(nn),      // %0
									"=r"(outptr0), // %1
									"=r"(outptr1), // %2
									"=r"(outptr2), // %3
									"=r"(outptr3), // %4
									"=r"(outptr4), // %5
									"=r"(outptr5), // %6
									"=r"(outptr6), // %7
									"=r"(outptr7), // %8
									"=r"(r0),      // %9
									"=r"(r1),      // %10
									"=r"(r2),      // %11
									"=r"(ktmp)     // %12
									: "0"(nn),
									"1"(outptr0),
									"2"(outptr1),
									"3"(outptr2),
									"4"(outptr3),
									"5"(outptr4),
									"6"(outptr5),
									"7"(outptr6),
									"8"(outptr7),
									"9"(r0),
									"10"(r1),
									"11"(r2),
									"12"(ktmp)
									: "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
							}
#else  // __aarch64__
							if (nn > 0)
							{
								asm volatile(
									"0:                             \n"
									"pld        [%1, #128]          \n"
									"vld1.s32   {d16-d17}, [%1]     \n" // out0
									"pld        [%2, #128]          \n"
									"vld1.s32   {d18-d19}, [%2]     \n" // out1
									"pld        [%3, #128]          \n"
									"vld1.s32   {d20-d21}, [%3]     \n" // out2
									"pld        [%4, #128]          \n"
									"vld1.s32   {d22-d23}, [%4]     \n" // out3

									// r0
									"pld        [%9, #64]          \n"
									"vld2.s8    {d8-d9}, [%9]       \n" // d8(a00 a02 a04 a06 a08 a010 a012 a014), d9(a01 a03 a05 a07 a09 a011 a013 a015)
									"add        %9, #8              \n"
									"pld        [%12, #64]         \n"
									"vld1.s8    {d0-d2}, [%12]!     \n" // d0(k00-k70) d1(k01-k71) d2(k02-k72)

									"pld        [%5, #128]          \n"
									"vld1.s32   {d24-d25}, [%5]     \n" // out4
									"pld        [%6, #128]          \n"
									"vld1.s32   {d26-d27}, [%6]     \n" // out5

									"vmovl.s8   q2, d2              \n" // q2(k02-k72)
									"vmovl.s8   q1, d1              \n" // q1(k01-k71)
									"vmovl.s8   q0, d0              \n" // q0(k00-k70)
									"vext.s8    d12, d8, d8, #1     \n" // d12(a02 a04 a06 a08 x x x x)

									"pld        [%7, #128]          \n"
									"vld1.s32   {d28-d29}, [%7]     \n" // out6

									"vmovl.s8   q5, d9              \n" // q5(a01 a03 a05 a07 a09 a011 a013 a015) d11
									"vmovl.s8   q4, d8              \n" // q4(a00 a02 a04 a06 a08 a010 a012 a014) d9
									"vmovl.s8   q6, d12             \n" // q6(a02 a04 a06 a08 a010 a012 a014 a016) d13

									"pld        [%8, #128]          \n"
									"vld1.s32   {d30-d31}, [%8]     \n" // out7

									"vmlal.s16  q8, d8, d0[0]       \n" // sum0 += (a00 a02 a04 a06) * k00
									"vmlal.s16  q9, d8, d0[1]       \n" // sum1 += (a00 a02 a04 a06) * k10
									"vmlal.s16  q10, d8, d0[2]      \n" // sum2 += (a00 a02 a04 a06) * k20
									"vmlal.s16  q11, d8, d0[3]      \n" // sum3 += (a00 a02 a04 a06) * k30
									"vmlal.s16  q12, d8, d1[0]      \n" // sum4 += (a00 a02 a04 a06) * k40
									"vmlal.s16  q13, d8, d1[1]      \n" // sum5 += (a00 a02 a04 a06) * k50
									"vmlal.s16  q14, d8, d1[2]      \n" // sum6 += (a00 a02 a04 a06) * k60
									"vmlal.s16  q15, d8, d1[3]      \n" // sum7 += (a00 a02 a04 a06) * k70

									"vmlal.s16  q8, d10, d2[0]      \n" // sum0 += (a01-a07) * k01
									"vmlal.s16  q9, d10, d2[1]      \n" // sum1 += (a01-a07) * k11
									"vmlal.s16  q10, d10, d2[2]     \n" // sum2 += (a01-a07) * k21
									"vmlal.s16  q11, d10, d2[3]     \n" // sum3 += (a01-a07) * k31
									"vmlal.s16  q12, d10, d3[0]     \n" // sum4 += (a01-a07) * k41
									"vmlal.s16  q13, d10, d3[1]     \n" // sum5 += (a01-a07) * k51
									"vmlal.s16  q14, d10, d3[2]     \n" // sum6 += (a01-a07) * k61
									"vmlal.s16  q15, d10, d3[3]     \n" // sum7 += (a01-a07) * k71

									"pld        [%10, #64]         \n"
									"vld2.s8    {d8-d9}, [%10]      \n" // d8(a10 a12 a14 a16 a18 a110 a112 a114), d9(a11 a13 a15 a17 a19 a111 a113 a115)
									"add        %10, #8             \n"

									"vmlal.s16  q8, d12, d4[0]      \n" // sum0 += (a02-a08) * k02
									"vmlal.s16  q9, d12, d4[1]      \n" // sum1 += (a02-a08) * k12
									"vmlal.s16  q10, d12, d4[2]     \n" // sum2 += (a02-a08) * k22
									"vmlal.s16  q11, d12, d4[3]     \n" // sum3 += (a02-a08) * k32

									"pld        [%12, #64]         \n"
									"vld1.s8    {d0-d2}, [%12]!     \n" // d0(k03-k73) d1(k04-k74) d2(k05-k75)

									"vmlal.s16  q12, d12, d5[0]     \n" // sum4 += (a02-a08) * k42
									"vmlal.s16  q13, d12, d5[1]     \n" // sum5 += (a02-a08) * k52
									"vmlal.s16  q14, d12, d5[2]     \n" // sum6 += (a02-a08) * k62
									"vmlal.s16  q15, d12, d5[3]     \n" // sum7 += (a02-a08) * k72

									// r1
									"vext.s8    d12, d8, d8, #1     \n" // d12(a12 a14 a16 a18 x x x x)

									"vmovl.s8   q2, d2              \n" // q2(k05-k75)
									"vmovl.s8   q1, d1              \n" // q1(k04-k74)
									"vmovl.s8   q0, d0              \n" // q0(k03-k73)
									"vmovl.s8   q5, d9              \n" // q5(a11-a115)
									"vmovl.s8   q4, d8              \n" // q4(a10-a114)
									"vmovl.s8   q6, d12             \n" // q6(a12-a116)

									"vmlal.s16  q8, d8, d0[0]       \n" // sum0 += (a10-a16) * k03
									"vmlal.s16  q9, d8, d0[1]       \n" // sum1 += (a10-a16) * k13
									"vmlal.s16  q10, d8, d0[2]      \n" // sum2 += (a10-a16) * k23
									"vmlal.s16  q11, d8, d0[3]      \n" // sum3 += (a10-a16) * k33
									"vmlal.s16  q12, d8, d1[0]      \n" // sum4 += (a10-a16) * k43
									"vmlal.s16  q13, d8, d1[1]      \n" // sum5 += (a10-a16) * k53
									"vmlal.s16  q14, d8, d1[2]      \n" // sum6 += (a10-a16) * k63
									"vmlal.s16  q15, d8, d1[3]      \n" // sum7 += (a10-a16) * k73

									"vmlal.s16  q8, d10, d2[0]      \n" // sum0 += (a11-a17) * k04
									"vmlal.s16  q9, d10, d2[1]      \n" // sum1 += (a11-a17) * k14
									"vmlal.s16  q10, d10, d2[2]     \n" // sum2 += (a11-a17) * k24
									"vmlal.s16  q11, d10, d2[3]     \n" // sum3 += (a11-a17) * k34
									"vmlal.s16  q12, d10, d3[0]     \n" // sum4 += (a11-a17) * k44
									"vmlal.s16  q13, d10, d3[1]     \n" // sum5 += (a11-a17) * k54
									"vmlal.s16  q14, d10, d3[2]     \n" // sum6 += (a11-a17) * k64
									"vmlal.s16  q15, d10, d3[3]     \n" // sum7 += (a11-a17) * k74

									"pld        [%11, #64]         \n"
									"vld2.s8    {d8-d9}, [%11]      \n" // d8(a20 a22 a24 a26 a28 a210 a212 a214), d9(a21 a23 a25 a27 a29 a211 a213 a215)
									"add        %11, #8             \n"

									"vmlal.s16  q8, d12, d4[0]      \n" // sum0 += (a12-a18) * k05
									"vmlal.s16  q9, d12, d4[1]      \n" // sum1 += (a12-a18) * k15
									"vmlal.s16  q10, d12, d4[2]     \n" // sum2 += (a12-a18) * k25
									"vmlal.s16  q11, d12, d4[3]     \n" // sum3 += (a12-a18) * k35

									"pld        [%12, #64]         \n"
									"vld1.s8    {d0-d2}, [%12]!     \n" // d0(k06-k76) d1(k07-k77) d2(k08-k78)

									"vmlal.s16  q12, d12, d5[0]     \n" // sum4 += (a12-a18) * k45
									"vmlal.s16  q13, d12, d5[1]     \n" // sum5 += (a12-a18) * k55
									"vmlal.s16  q14, d12, d5[2]     \n" // sum6 += (a12-a18) * k65
									"vmlal.s16  q15, d12, d5[3]     \n" // sum7 += (a12-a18) * k75

									// r2
									"vext.s8    d12, d8, d8, #1     \n" // d12(a22 a24 a26 a28 x x x x)

									"vmovl.s8   q2, d2              \n" // q2(k08-k78)
									"vmovl.s8   q1, d1              \n" // q1(k07-k77)
									"vmovl.s8   q0, d0              \n" // q0(k06-k76)
									"vmovl.s8   q5, d9              \n" // q5(a21-a215)
									"vmovl.s8   q4, d8              \n" // q4(a20-a214)
									"vmovl.s8   q6, d12             \n" // q6(a22-a216)

									"vmlal.s16  q8, d8, d0[0]       \n" // sum0 += (a20-a26) * k06
									"vmlal.s16  q9, d8, d0[1]       \n" // sum1 += (a20-a26) * k16
									"vmlal.s16  q10, d8, d0[2]      \n" // sum2 += (a20-a26) * k26
									"vmlal.s16  q11, d8, d0[3]      \n" // sum3 += (a20-a26) * k36
									"vmlal.s16  q12, d8, d1[0]      \n" // sum4 += (a20-a26) * k46
									"vmlal.s16  q13, d8, d1[1]      \n" // sum5 += (a20-a26) * k56
									"vmlal.s16  q14, d8, d1[2]      \n" // sum6 += (a20-a26) * k66
									"vmlal.s16  q15, d8, d1[3]      \n" // sum7 += (a20-a26) * k76

									"vmlal.s16  q8, d10, d2[0]      \n" // sum0 += (a21-a27) * k07
									"vmlal.s16  q9, d10, d2[1]      \n" // sum1 += (a21-a27) * k17
									"vmlal.s16  q10, d10, d2[2]     \n" // sum2 += (a21-a27) * k27
									"vmlal.s16  q11, d10, d2[3]     \n" // sum3 += (a21-a27) * k37
									"vmlal.s16  q12, d10, d3[0]     \n" // sum4 += (a21-a27) * k47
									"vmlal.s16  q13, d10, d3[1]     \n" // sum5 += (a21-a27) * k57
									"vmlal.s16  q14, d10, d3[2]     \n" // sum6 += (a21-a27) * k67
									"vmlal.s16  q15, d10, d3[3]     \n" // sum7 += (a21-a27) * k77

									"vmlal.s16  q8, d12, d4[0]      \n" // sum0 += (a22-a28) * k08
									"vmlal.s16  q9, d12, d4[1]      \n" // sum1 += (a22-a28) * k18
									"vmlal.s16  q10, d12, d4[2]     \n" // sum2 += (a22-a28) * k28
									"vmlal.s16  q11, d12, d4[3]     \n" // sum3 += (a22-a28) * k38
									"vmlal.s16  q12, d12, d5[0]     \n" // sum4 += (a22-a28) * k48
									"vmlal.s16  q13, d12, d5[1]     \n" // sum5 += (a22-a28) * k58
									"vmlal.s16  q14, d12, d5[2]     \n" // sum6 += (a22-a28) * k68
									"vmlal.s16  q15, d12, d5[3]     \n" // sum7 += (a22-a28) * k78

									// save s32 to memory
									"sub        %12, %12, #72       \n"
									"vst1.s32   {d16-d17}, [%1]!    \n" // out0
									"vst1.s32   {d18-d19}, [%2]!    \n" // out1
									"vst1.s32   {d20-d21}, [%3]!    \n" // out2
									"vst1.s32   {d22-d23}, [%4]!    \n" // out3
									"subs       %0, #1              \n"
									"vst1.s32   {d24-d25}, [%5]!    \n" // out4
									"vst1.s32   {d26-d27}, [%6]!    \n" // out5
									"vst1.s32   {d28-d29}, [%7]!    \n" // out6
									"vst1.s32   {d30-d31}, [%8]!    \n" // out7

									"bne        0b                  \n"
									: "=r"(nn),      // %0
									"=r"(outptr0), // %1
									"=r"(outptr1), // %2
									"=r"(outptr2), // %3
									"=r"(outptr3), // %4
									"=r"(outptr4), // %5
									"=r"(outptr5), // %6
									"=r"(outptr6), // %7
									"=r"(outptr7), // %8
									"=r"(r0),      // %9
									"=r"(r1),      // %10
									"=r"(r2),      // %11
									"=r"(ktmp)     // %12
									: "0"(nn),
									"1"(outptr0),
									"2"(outptr1),
									"3"(outptr2),
									"4"(outptr3),
									"5"(outptr4),
									"6"(outptr5),
									"7"(outptr6),
									"8"(outptr7),
									"9"(r0),
									"10"(r1),
									"11"(r2),
									"12"(ktmp)
									: "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
							}
#endif // __aarch64__
#endif // __ARM_NEON
							for (; remain > 0; remain--)
							{
#if __ARM_NEON
#if __aarch64__
								int8x8_t _r0_s8 = vld1_s8(r0); // (a00 a01 a02 ....)
								int8x8_t _r1_s8 = vld1_s8(r1); // (a10 a11 a12 ....)
								int8x8_t _r2_s8 = vld1_s8(r2); // (a20 a21 a22 ....)

								int16x8_t _r0 = vmovl_s8(_r0_s8);
								int16x8_t _r1 = vmovl_s8(_r1_s8);
								int16x8_t _r2 = vmovl_s8(_r2_s8);

								int32x4_t _sum03 = {};
								int32x4_t _sum47 = {};

								_sum03 = vld1q_lane_s32(outptr0, _sum03, 0); // out0
								_sum03 = vld1q_lane_s32(outptr1, _sum03, 1); // out1
								_sum03 = vld1q_lane_s32(outptr2, _sum03, 2); // out2
								_sum03 = vld1q_lane_s32(outptr3, _sum03, 3); // out3
								_sum47 = vld1q_lane_s32(outptr4, _sum47, 0); // out4
								_sum47 = vld1q_lane_s32(outptr5, _sum47, 1); // out5
								_sum47 = vld1q_lane_s32(outptr6, _sum47, 2); // out6
								_sum47 = vld1q_lane_s32(outptr7, _sum47, 3); // out7

								// k0 - k2
								int8x8_t _k0_8 = vld1_s8(ktmp);      //(k00-k70)
								int8x8_t _k1_8 = vld1_s8(ktmp + 8);  //(k01-k71)
								int8x8_t _k2_8 = vld1_s8(ktmp + 16); //(k02-k72)

								int16x8_t _k0 = vmovl_s8(_k0_8);
								int16x8_t _k1 = vmovl_s8(_k1_8);
								int16x8_t _k2 = vmovl_s8(_k2_8);

								int32x4_t _sum0 = vmull_laneq_s16(vget_low_s16(_k0), _r0, 0);
								int32x4_t _sum0n = vmull_laneq_s16(vget_high_s16(_k0), _r0, 0);
								int32x4_t _sum1 = vmull_laneq_s16(vget_low_s16(_k1), _r0, 1);
								int32x4_t _sum1n = vmull_laneq_s16(vget_high_s16(_k1), _r0, 1);
								_sum03 = vmlal_laneq_s16(_sum03, vget_low_s16(_k2), _r0, 2);
								_sum47 = vmlal_laneq_s16(_sum47, vget_high_s16(_k2), _r0, 2);

								// k3 - k5
								_k0_8 = vld1_s8(ktmp + 24); //(k03-k73)
								_k1_8 = vld1_s8(ktmp + 32); //(k04-k74)
								_k2_8 = vld1_s8(ktmp + 40); //(k05-k75)

								_k0 = vmovl_s8(_k0_8);
								_k1 = vmovl_s8(_k1_8);
								_k2 = vmovl_s8(_k2_8);

								_sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_k0), _r1, 0);
								_sum0n = vmlal_laneq_s16(_sum0n, vget_high_s16(_k0), _r1, 0);
								_sum1 = vmlal_laneq_s16(_sum1, vget_low_s16(_k1), _r1, 1);
								_sum1n = vmlal_laneq_s16(_sum1n, vget_high_s16(_k1), _r1, 1);
								_sum03 = vmlal_laneq_s16(_sum03, vget_low_s16(_k2), _r1, 2);
								_sum47 = vmlal_laneq_s16(_sum47, vget_high_s16(_k2), _r1, 2);

								// k6 - k8
								_k0_8 = vld1_s8(ktmp + 48); //(k06-k76)
								_k1_8 = vld1_s8(ktmp + 56); //(k07-k77)
								_k2_8 = vld1_s8(ktmp + 64); //(k08-k78)

								_k0 = vmovl_s8(_k0_8);
								_k1 = vmovl_s8(_k1_8);
								_k2 = vmovl_s8(_k2_8);

								_sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_k0), _r2, 0);
								_sum0n = vmlal_laneq_s16(_sum0n, vget_high_s16(_k0), _r2, 0);
								_sum1 = vmlal_laneq_s16(_sum1, vget_low_s16(_k1), _r2, 1);
								_sum1n = vmlal_laneq_s16(_sum1n, vget_high_s16(_k1), _r2, 1);
								_sum03 = vmlal_laneq_s16(_sum03, vget_low_s16(_k2), _r2, 2);
								_sum47 = vmlal_laneq_s16(_sum47, vget_high_s16(_k2), _r2, 2);

								_sum0 = vaddq_s32(_sum0, _sum1);
								_sum0n = vaddq_s32(_sum0n, _sum1n);
								_sum03 = vaddq_s32(_sum03, _sum0);
								_sum47 = vaddq_s32(_sum47, _sum0n);

								vst1q_lane_s32(outptr0, _sum03, 0);
								vst1q_lane_s32(outptr1, _sum03, 1);
								vst1q_lane_s32(outptr2, _sum03, 2);
								vst1q_lane_s32(outptr3, _sum03, 3);
								vst1q_lane_s32(outptr4, _sum47, 0);
								vst1q_lane_s32(outptr5, _sum47, 1);
								vst1q_lane_s32(outptr6, _sum47, 2);
								vst1q_lane_s32(outptr7, _sum47, 3);

								outptr0++;
								outptr1++;
								outptr2++;
								outptr3++;
								outptr4++;
								outptr5++;
								outptr6++;
								outptr7++;
#else  // __aarch64__
								asm volatile(
									"pld        [%8, #64]          \n"
									"vld1.s8    {d0}, [%8]         \n" // d0(a00 a01 a02 ....)
									"pld        [%9, #64]          \n"
									"vld1.s8    {d2}, [%9]         \n" // d2(a10 a11 a12 ....)
									"pld        [%10, #64]         \n"
									"vld1.s8    {d4}, [%10]        \n" // d4(a20 a21 a22 ....)

									"pld        [%11, #64]         \n"
									"vld1.s8    {d6-d8}, [%11]!    \n" // d6(k00-k70) d7(k01-k71) d8(k02-k72)

									"vmovl.s8   q0, d0             \n" // d0(a00 a01 a02 x)
									"vmovl.s8   q1, d2             \n" // d2(a10 a11 a12 x)
									"vmovl.s8   q2, d4             \n" // d4(a20 a21 a22 x)

									"vmovl.s8   q5, d8             \n" // d10(k02-k32) d11(k42-k72)
									"vmovl.s8   q4, d7             \n" // d8(k01-k31) d9(k41-k71)
									"vmovl.s8   q3, d6             \n" // d6(k00-k30) d7(k40-k70)

									"vld1.s32   {d20[0]}, [%0]     \n" // out0 q10
									"vld1.s32   {d20[1]}, [%1]     \n" // out1
									"vld1.s32   {d21[0]}, [%2]     \n" // out2
									"vld1.s32   {d21[1]}, [%3]     \n" // out3

									"pld        [%11, #64]         \n"
									"vld1.s8    {d24-d26}, [%11]!  \n"
									"vmovl.s8   q14, d26           \n" // d28(k05-k35) d29(k45-k75)
									"vmovl.s8   q13, d25           \n" // d26(k04-k34) d27(k44-k74)
									"vmovl.s8   q12, d24           \n" // d24(k03-k33) d25(k43-k73)

									"vld1.s32   {d22[0]}, [%4]     \n" // out4 q11
									"vld1.s32   {d22[1]}, [%5]     \n" // out5
									"vld1.s32   {d23[0]}, [%6]     \n" // out6
									"vld1.s32   {d23[1]}, [%7]     \n" // out7

									"vmull.s16  q6, d6, d0[0]      \n" // a00 x (k00-k30)
									"vmull.s16  q7, d7, d0[0]      \n" // a00 x (k40-k70)
									"vmull.s16  q8, d8, d0[1]      \n" // a01 x (k01-k31)
									"vmull.s16  q9, d9, d0[1]      \n" // a01 x (k41-k71)
									"vmlal.s16  q10, d10, d0[2]    \n" // a02 x (k02-k32)
									"vmlal.s16  q11, d11, d0[2]    \n" // a02 x (k42-k72)

									"pld        [%11, #64]         \n"
									"vld1.s8    {d6-d8}, [%11]!    \n"
									"vmovl.s8   q5, d8             \n" // d10(k08-k38) d11(k48-k78)
									"vmovl.s8   q4, d7             \n" // d8(k07-k37) d9(k47-k77)
									"vmovl.s8   q3, d6             \n" // d6(k06-k36) d7(k46-k76)

									"vmlal.s16  q6, d24, d2[0]     \n" // a10 x (k03-k33)
									"vmlal.s16  q7, d25, d2[0]     \n" // a10 x (k43-k73)
									"vmlal.s16  q8, d26, d2[1]     \n" // a11 x (k04-k34)
									"vmlal.s16  q9, d27, d2[1]     \n" // a11 x (k44-k74)
									"vmlal.s16  q10, d28, d2[2]    \n" // a12 x (k05-k35)
									"vmlal.s16  q11, d29, d2[2]    \n" // a12 x (k45-k75)

									"vmlal.s16  q6, d6, d4[0]      \n" // a20 x (k06-k36)
									"vmlal.s16  q7, d7, d4[0]      \n" // a20 x (k46-k76)
									"vmlal.s16  q8, d8, d4[1]      \n" // a21 x (k07-k37)
									"vmlal.s16  q9, d9, d4[1]      \n" // a21 x (k47-k77)
									"vmlal.s16  q10, d10, d4[2]    \n" // a22 x (k08-k38)
									"vmlal.s16  q11, d11, d4[2]    \n" // a22 x (k48-k78)

									"vadd.s32   q8, q8, q6         \n"
									"vadd.s32   q9, q9, q7         \n"

									"sub        %11, %11, #72      \n"

									"vadd.s32   q10, q10, q8       \n"
									"vadd.s32   q11, q11, q9       \n"

									"vst1.s32   {d20[0]}, [%0]!    \n" // out0
									"vst1.s32   {d20[1]}, [%1]!    \n" // out1
									"vst1.s32   {d21[0]}, [%2]!    \n" // out2
									"vst1.s32   {d21[1]}, [%3]!    \n" // out3
									"vst1.s32   {d22[0]}, [%4]!    \n" // out4
									"vst1.s32   {d22[1]}, [%5]!    \n" // out5
									"vst1.s32   {d23[0]}, [%6]!    \n" // out6
									"vst1.s32   {d23[1]}, [%7]!    \n" // out7

									: "=r"(outptr0), // %0
									"=r"(outptr1), // %1
									"=r"(outptr2), // %2
									"=r"(outptr3), // %3
									"=r"(outptr4), // %4
									"=r"(outptr5), // %5
									"=r"(outptr6), // %6
									"=r"(outptr7), // %7
									"=r"(r0),      // %8
									"=r"(r1),      // %9
									"=r"(r2),      // %10
									"=r"(ktmp)     // %11
									: "0"(outptr0),
									"1"(outptr1),
									"2"(outptr2),
									"3"(outptr3),
									"4"(outptr4),
									"5"(outptr5),
									"6"(outptr6),
									"7"(outptr7),
									"8"(r0),
									"9"(r1),
									"10"(r2),
									"11"(ktmp)
									: "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else  // __ARM_NEON
								int sum0 = 0;
								int sum1 = 0;
								int sum2 = 0;
								int sum3 = 0;
								int sum4 = 0;
								int sum5 = 0;
								int sum6 = 0;
								int sum7 = 0;

								sum0 += (int)r0[0] * ktmp[0];
								sum1 += (int)r0[0] * ktmp[1];
								sum2 += (int)r0[0] * ktmp[2];
								sum3 += (int)r0[0] * ktmp[3];
								sum4 += (int)r0[0] * ktmp[4];
								sum5 += (int)r0[0] * ktmp[5];
								sum6 += (int)r0[0] * ktmp[6];
								sum7 += (int)r0[0] * ktmp[7];
								ktmp += 8;

								sum0 += (int)r0[1] * ktmp[0];
								sum1 += (int)r0[1] * ktmp[1];
								sum2 += (int)r0[1] * ktmp[2];
								sum3 += (int)r0[1] * ktmp[3];
								sum4 += (int)r0[1] * ktmp[4];
								sum5 += (int)r0[1] * ktmp[5];
								sum6 += (int)r0[1] * ktmp[6];
								sum7 += (int)r0[1] * ktmp[7];
								ktmp += 8;

								sum0 += (int)r0[2] * ktmp[0];
								sum1 += (int)r0[2] * ktmp[1];
								sum2 += (int)r0[2] * ktmp[2];
								sum3 += (int)r0[2] * ktmp[3];
								sum4 += (int)r0[2] * ktmp[4];
								sum5 += (int)r0[2] * ktmp[5];
								sum6 += (int)r0[2] * ktmp[6];
								sum7 += (int)r0[2] * ktmp[7];
								ktmp += 8;

								sum0 += (int)r1[0] * ktmp[0];
								sum1 += (int)r1[0] * ktmp[1];
								sum2 += (int)r1[0] * ktmp[2];
								sum3 += (int)r1[0] * ktmp[3];
								sum4 += (int)r1[0] * ktmp[4];
								sum5 += (int)r1[0] * ktmp[5];
								sum6 += (int)r1[0] * ktmp[6];
								sum7 += (int)r1[0] * ktmp[7];
								ktmp += 8;

								sum0 += (int)r1[1] * ktmp[0];
								sum1 += (int)r1[1] * ktmp[1];
								sum2 += (int)r1[1] * ktmp[2];
								sum3 += (int)r1[1] * ktmp[3];
								sum4 += (int)r1[1] * ktmp[4];
								sum5 += (int)r1[1] * ktmp[5];
								sum6 += (int)r1[1] * ktmp[6];
								sum7 += (int)r1[1] * ktmp[7];
								ktmp += 8;

								sum0 += (int)r1[2] * ktmp[0];
								sum1 += (int)r1[2] * ktmp[1];
								sum2 += (int)r1[2] * ktmp[2];
								sum3 += (int)r1[2] * ktmp[3];
								sum4 += (int)r1[2] * ktmp[4];
								sum5 += (int)r1[2] * ktmp[5];
								sum6 += (int)r1[2] * ktmp[6];
								sum7 += (int)r1[2] * ktmp[7];
								ktmp += 8;

								sum0 += (int)r2[0] * ktmp[0];
								sum1 += (int)r2[0] * ktmp[1];
								sum2 += (int)r2[0] * ktmp[2];
								sum3 += (int)r2[0] * ktmp[3];
								sum4 += (int)r2[0] * ktmp[4];
								sum5 += (int)r2[0] * ktmp[5];
								sum6 += (int)r2[0] * ktmp[6];
								sum7 += (int)r2[0] * ktmp[7];
								ktmp += 8;

								sum0 += (int)r2[1] * ktmp[0];
								sum1 += (int)r2[1] * ktmp[1];
								sum2 += (int)r2[1] * ktmp[2];
								sum3 += (int)r2[1] * ktmp[3];
								sum4 += (int)r2[1] * ktmp[4];
								sum5 += (int)r2[1] * ktmp[5];
								sum6 += (int)r2[1] * ktmp[6];
								sum7 += (int)r2[1] * ktmp[7];
								ktmp += 8;

								sum0 += (int)r2[2] * ktmp[0];
								sum1 += (int)r2[2] * ktmp[1];
								sum2 += (int)r2[2] * ktmp[2];
								sum3 += (int)r2[2] * ktmp[3];
								sum4 += (int)r2[2] * ktmp[4];
								sum5 += (int)r2[2] * ktmp[5];
								sum6 += (int)r2[2] * ktmp[6];
								sum7 += (int)r2[2] * ktmp[7];
								ktmp += 8;

								*outptr0 += sum0;
								*outptr1 += sum1;
								*outptr2 += sum2;
								*outptr3 += sum3;
								*outptr4 += sum4;
								*outptr5 += sum5;
								*outptr6 += sum6;
								*outptr7 += sum7;

								ktmp -= 8 * 9;

								outptr0++;
								outptr1++;
								outptr2++;
								outptr3++;
								outptr4++;
								outptr5++;
								outptr6++;
								outptr7++;
#endif // __ARM_NEON
								r0 += 2;
								r1 += 2;
								r2 += 2;
							}

							r0 += tailstep;
							r1 += tailstep;
							r2 += tailstep;
						}

						ktmp += 8 * 9;
					}
				}

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
				for (int p = remain_outch_start; p < outch; p++)
				{
					int* out = top_data + p * top_cstep;

					fill(out, top_cstep, 0);

					const signed char* ktmp = kernel_tm_int8_data + (p / 8 + p % 8) * kernel_tm_int8_cstep;

					for (int q = 0; q < inch; q++)
					{
						int* outptr = out;

						const signed char* img0 = bottom_data + q * bottom_cstep;

						const signed char* r0 = img0;
						const signed char* r1 = img0 + w;
						const signed char* r2 = img0 + w * 2;

						int i = 0;

						for (; i < outh; i++)
						{
#if __ARM_NEON
							int nn = outw >> 3;
							int remain = outw & 7;
#else
							int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
							if (nn > 0)
							{
								asm volatile(
									"0:                                   \n"

									"ld1    {v0.8b, v1.8b}, [%5]          \n" //ktmp
									"ld2    {v2.8b, v3.8b}, [%2], #16     \n" //r0-r2
									"ld2    {v4.8b, v5.8b}, [%2]          \n"

									"ld2    {v6.8b, v7.8b}, [%3], #16     \n" //r3-r5
									"ld2    {v8.8b, v9.8b}, [%3]          \n"

									"ld2    {v10.8b, v11.8b}, [%4], #16   \n" //r6-r8
									"ld2    {v12.8b, v13.8b}, [%4]        \n"

									"ld1    {v14.4s, v15.4s}, [%1]        \n" //out0

									"ext    v4.8b, v2.8b, v4.8b, #1       \n"
									"ext    v8.8b, v6.8b, v8.8b, #1       \n"
									"ext    v12.8b, v10.8b, v12.8b, #1    \n"

									"sshll  v0.8h, v0.8b, #0              \n" //(k0-k7)
									"sshll  v1.8h, v1.8b, #0              \n" //(k8)
									"sshll  v2.8h, v2.8b, #0              \n" // r0
									"sshll  v3.8h, v3.8b, #0              \n" // r1
									"sshll  v4.8h, v4.8b, #0              \n" // r2
									"sshll  v6.8h, v6.8b, #0              \n" // r3
									"sshll  v7.8h, v7.8b, #0              \n" // r4
									"sshll  v8.8h, v8.8b, #0              \n" // r5
									"sshll  v10.8h, v10.8b, #0            \n" // r6
									"sshll  v11.8h, v11.8b, #0            \n" // r7
									"sshll  v12.8h, v12.8b, #0            \n" // r8

									// r0
									"smull  v16.4s, v2.4h, v0.h[0]        \n" // out = r0*k0
									"smull2  v17.4s, v2.8h, v0.h[0]       \n"
									"smull  v18.4s, v3.4h, v0.h[1]        \n" // outn = r1*k1
									"smull2  v19.4s, v3.8h, v0.h[1]       \n"
									"smlal  v16.4s, v4.4h, v0.h[2]        \n" // out = r2*k2
									"smlal2  v17.4s, v4.8h, v0.h[2]       \n"
									"smlal  v18.4s, v6.4h, v0.h[3]        \n" // outn = r3*k3
									"smlal2  v19.4s, v6.8h, v0.h[3]       \n"
									"smlal  v16.4s, v7.4h, v0.h[4]        \n" // out = r4*k4
									"smlal2  v17.4s, v7.8h, v0.h[4]       \n"
									"smlal  v18.4s, v8.4h, v0.h[5]        \n" // outn = r5*k5
									"smlal2  v19.4s, v8.8h, v0.h[5]       \n"
									"smlal  v16.4s, v10.4h, v0.h[6]       \n" // out = r6*k6
									"smlal2  v17.4s, v10.8h, v0.h[6]      \n"
									"smlal  v18.4s, v11.4h, v0.h[7]       \n" // outn = r7*k7
									"smlal2  v19.4s, v11.8h, v0.h[7]      \n"
									"smlal  v16.4s, v12.4h, v1.h[0]       \n" // out = r8*k8
									"smlal2  v17.4s, v12.8h, v1.h[0]      \n"

									"add    v8.4s, v16.4s, v18.4s         \n"
									"add    v9.4s, v17.4s, v19.4s         \n"

									"st1    {v8.4s, v9.4s}, [%1], #32     \n"

									"subs   %w0, %w0, #1                  \n"

									"bne    0b                            \n"

									: "=r"(nn),     // %0
									"=r"(outptr), // %1
									"=r"(r0),     // %2
									"=r"(r1),     // %3
									"=r"(r2),     // %4
									"=r"(ktmp)    // %5
									: "0"(nn),
									"1"(outptr),
									"2"(r0),
									"3"(r1),
									"4"(r2),
									"5"(ktmp)
									: "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
							}
#else
							if (nn > 0)
							{
								asm volatile(
									"vld1.s8    {d0-d1}, [%5]       \n" // d0(k0 - k7) d1(k8 ...)
									"vmovl.s8   q1, d1              \n" // d2(k8 ...)
									"vmovl.s8   q0, d0              \n" // d0(k0 - k3) d1(k4 - k7)
									"0:                             \n"
									"pld        [%2, #192]          \n"
									"vld2.s8    {d4-d5}, [%2]!      \n" // r0 d4(a00 a02 ... a014) d5(a01 a03 ... a015)
									"vld2.s8    {d8-d9}, [%2]       \n" //    d8(a016 ....)
									"vld2.s8    {d10-d11}, [%3]!    \n" // r1 d10(a10 a12 ... a114) d11(a11 a13 ... a115)
									"vld2.s8    {d14-d15}, [%3]     \n" //    d14(a116 ....)
									"vld2.s8    {d16-d17}, [%4]!    \n" // r2 d16(a20 a22 ... a214) d17(a21 a23 ... a215)
									"vld2.s8    {d20-d21}, [%4]     \n" //    d20(a216 ....)
									"vld1.s32   {d22-d25}, [%1]     \n" // q11(out0 - out3) q12(out4 - out7)

									"vext.s8    d8, d4, d8, #1      \n" //  d8(a02 a04 ... a016)
									"vext.s8    d14, d10, d14, #1   \n" // d14(a12 a14 ... a116)
									"vext.s8    d20, d16, d20, #1   \n" // d20(a22 a24 ... a216)

									"vmovl.s8   q3, d5              \n" // q3(a01 a03 ... a015)
									"vmovl.s8   q2, d4              \n" // q2(a00 a02 ... a014)
									"vmovl.s8   q4, d8              \n" // q4(a02 a04 ... a016)

									"vmovl.s8   q6, d11             \n" // q6(a11 a13 ... a115)
									"vmovl.s8   q5, d10             \n" // q5(a10 a12 ... a114)
									"vmovl.s8   q7, d14             \n" // q7(a12 a14 ... a116)

									"vmovl.s8   q9, d17             \n" // q9(a21 a23 ... a215)
									"vmovl.s8   q8, d16             \n" // q8(a20 a22 ... a214)
									"vmovl.s8   q10, d20            \n" // q10(a22 a24 ... a216)

									"vmlal.s16  q11, d4, d0[0]      \n" // k0
									"vmlal.s16  q12, d5, d0[0]      \n"
									"vmull.s16  q13, d6, d0[1]      \n" // k1
									"vmull.s16  q14, d7, d0[1]      \n"
									"vmlal.s16  q11, d8, d0[2]      \n" // k2
									"vmlal.s16  q12, d9, d0[2]      \n"

									"vmlal.s16  q13, d12, d1[0]     \n" // k4
									"vmlal.s16  q14, d13, d1[0]     \n"
									"vmlal.s16  q11, d10, d0[3]     \n" // k3
									"vmlal.s16  q12, d11, d0[3]     \n"
									"vmlal.s16  q13, d14, d1[1]     \n" // k5
									"vmlal.s16  q14, d15, d1[1]     \n"

									"vmlal.s16  q11, d16, d1[2]     \n" // k6
									"vmlal.s16  q12, d17, d1[2]     \n"
									"vmlal.s16  q13, d18, d1[3]     \n" // k7
									"vmlal.s16  q14, d19, d1[3]     \n"
									"vmlal.s16  q11, d20, d2[0]     \n" // k8
									"vmlal.s16  q12, d21, d2[0]     \n"

									"vadd.s32   q11, q11, q13       \n"
									"vadd.s32   q12, q12, q14       \n"

									"vst1.32    {d22-d25}, [%1]!    \n"

									"subs       %0, #1              \n"
									"bne        0b                  \n"
									: "=r"(nn),     // %0
									"=r"(outptr), // %1
									"=r"(r0),     // %2
									"=r"(r1),     // %3
									"=r"(r2),     // %4
									"=r"(ktmp)    // %5
									: "0"(nn),
									"1"(outptr),
									"2"(r0),
									"3"(r1),
									"4"(r2),
									"5"(ktmp)
									: "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
							}
#endif // __aarch64__
#endif // __ARM_NEON
							if (remain > 0)
							{
#if __ARM_NEON
								int8x8_t _k01234567s8 = vld1_s8(ktmp);
								int8x8_t _k8xxxxxxxs8 = vld1_s8(ktmp + 8);
								int8x8_t _k34567xxxs8 = vext_s8(_k01234567s8, _k01234567s8, 3);
								int8x8_t _k678xxxxxs8 = vext_s8(_k01234567s8, _k8xxxxxxxs8, 6);
								int16x8_t _k0123_s16 = vmovl_s8(_k01234567s8);
								int16x8_t _k3456_s16 = vmovl_s8(_k34567xxxs8);
								int16x8_t _k678x_s16 = vmovl_s8(_k678xxxxxs8);
#endif
								for (; remain > 0; remain--)
								{
#if __ARM_NEON
									int8x8_t _r00s8 = vld1_s8(r0);
									int8x8_t _r10s8 = vld1_s8(r1);
									int8x8_t _r20s8 = vld1_s8(r2);

									int16x8_t _r00s16 = vmovl_s8(_r00s8);
									int16x8_t _r10s16 = vmovl_s8(_r10s8);
									int16x8_t _r20s16 = vmovl_s8(_r20s8);

									int32x4_t _sum = vmull_s16(vget_low_s16(_r00s16), vget_low_s16(_k0123_s16));
									_sum = vmlal_s16(_sum, vget_low_s16(_r10s16), vget_low_s16(_k3456_s16));
									_sum = vmlal_s16(_sum, vget_low_s16(_r20s16), vget_low_s16(_k678x_s16));

									_sum = vsetq_lane_s32(*outptr, _sum, 3);

#if __aarch64__
									* outptr = vaddvq_s32(_sum);
#else
									int32x2_t _ss = vadd_s32(vget_low_s32(_sum), vget_high_s32(_sum));
									_ss = vpadd_s32(_ss, _ss);

									*outptr = vget_lane_s32(_ss, 0);
#endif // __aarch64__
#else
									int sum = 0;

									sum += (int)r0[0] * ktmp[0];
									sum += (int)r0[1] * ktmp[1];
									sum += (int)r0[2] * ktmp[2];
									sum += (int)r1[0] * ktmp[3];
									sum += (int)r1[1] * ktmp[4];
									sum += (int)r1[2] * ktmp[5];
									sum += (int)r2[0] * ktmp[6];
									sum += (int)r2[1] * ktmp[7];
									sum += (int)r2[2] * ktmp[8];

									*outptr += sum;
#endif // __ARM_NEON
									r0 += 2;
									r1 += 2;
									r2 += 2;
									outptr++;
								}
							}

							r0 += tailstep;
							r1 += tailstep;
							r2 += tailstep;
						}

						ktmp += 9;
					}
				}
			}
			
		}

#if __aarch64__
		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv1x1s1_sgemm_int8_neon(const std::shared_ptr<memory::tensor<int8_t>>& bottom,
			std::shared_ptr<memory::tensor<int>>& top)
		{
			const int num = bottom->num();
			const int n = bottom->count(2, 4);
			const int k = bottom->channels();
			const int m = top->channels();
			const int ldc = top->count(2, 4);

			const int8_t* kernel_tm_int8_data = kernel_tm_int8_->cpu_data();

			memory::tensor<int8_t> bottom_tm(k * n, this->params_.device_, memory::NCHW);
			const int8_t* bottom_tm_data = bottom_tm.cpu_data();
			for (size_t num_i = 0; num_i < num; num_i++)
			{
				{
					const int8_t* pData = bottom->cpu_data() + num_i * k * n;
					int8_t* pReorder = bottom_tm.mutable_cpu_data();
					reorder_b(pData, pReorder, k, n, n);
				}

				// GEMM
				int* pc = top->mutable_cpu_data() + num_i * m * ldc;
				const int8_t* pa = kernel_tm_int8_data;
				const int8_t* pb = bottom_tm_data;

				int8kernel((void*)pc, pa, pb, m, k, n, ldc, 0, 0);
			}
		}
#else
		template<typename Dtype>
		void operation_convolution_arm<Dtype>::conv1x1s1_sgemm_int8_neon(const std::shared_ptr<memory::tensor<int8_t>>& bottom,
			std::shared_ptr<memory::tensor<int>>& top)
		{
			int num = bottom->num();
			int w = bottom->width();
			int h = bottom->height();
			int inch = bottom->channels();
			int outch = top->channels();

			const int size = w * h;

			int bottom_cstep = w * h;
			int top_cstep = top->count(2, 4);
			const int8_t* kernel_tm_int8_data = kernel_tm_int8_->cpu_data();
			int kernel_tm_int8_cstep = kernel_tm_int8_->count(2, 4);

			// interleave
			memory::tensor<int8_t> tmp(std::vector<int>{1, size / 8 + (size % 8) / 4 + size % 4, inch / 4 + inch % 4, 8 * 4}, this->params_.device_, memory::NCHW);
			int8_t* tmp_data = tmp.mutable_cpu_data();
			int tmp_cstep = tmp.count(2, 4);
			for (int num_i = 0; num_i < num; num_i++)
			{
				const int8_t* bottom_data = bottom->cpu_data() + num_i * inch * bottom_cstep;
				int* top_data = top->mutable_cpu_data() + num_i * outch * top_cstep;

				{
					int nn_size = size >> 3;
					int remain_size_start = nn_size << 3;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int ii = 0; ii < nn_size; ii++)
					{
						int i = ii * 8;

						const signed char* img0 = bottom_data;
						img0 += i;

						signed char* tmpptr = tmp_data + (i / 8) * tmp_cstep;

						for (int q = 0; q < inch; q++)
						{
#if __ARM_NEON
							asm volatile(
								"pld        [%0, #64]     \n"
								"vld1.s8   {d0}, [%0]     \n"
								"vst1.s8   {d0}, [%1]!    \n"
								: "=r"(img0),  // %0
								"=r"(tmpptr) // %1
								: "0"(img0),
								"1"(tmpptr)
								: "memory", "d0");
							img0 += bottom_cstep;
#else
							tmpptr[0] = img0[0];
							tmpptr[1] = img0[1];
							tmpptr[2] = img0[2];
							tmpptr[3] = img0[3];
							tmpptr[4] = img0[4];
							tmpptr[5] = img0[5];
							tmpptr[6] = img0[6];
							tmpptr[7] = img0[7];

							tmpptr += 8;
							img0 += bottom_cstep;
#endif // __ARM_NEON
						}
					}

					nn_size = (size - remain_size_start) >> 2;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int ii = 0; ii < nn_size; ii++)
					{
						int i = remain_size_start + ii * 4;

						const signed char* img0 = bottom_data;
						img0 += i;

						signed char* tmpptr = tmp_data + (i / 8 + (i % 8) / 4)* tmp_cstep;

						for (int q = 0; q < inch; q++)
						{
							tmpptr[0] = img0[0];
							tmpptr[1] = img0[1];
							tmpptr[2] = img0[2];
							tmpptr[3] = img0[3];

							tmpptr += 4;
							img0 += bottom_cstep;
						}
					}

					remain_size_start += nn_size << 2;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int i = remain_size_start; i < size; i++)
					{
						const signed char* img0 = bottom_data;
						img0 += i;

						signed char* tmpptr = tmp_data + (i / 8 + (i % 8) / 4 + i % 4) * tmp_cstep;

						for (int q = 0; q < inch; q++)
						{
							tmpptr[0] = img0[0];
							tmpptr++;
							img0 += bottom_cstep;
						}
					}
				}

				// sgemm process
				int nn_outch = 0;
				int remain_outch_start = 0;

				nn_outch = (outch - remain_outch_start) >> 2;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
				for (int pp = 0; pp < nn_outch; pp++)
				{
					int p = remain_outch_start + pp * 4;

					int* outptr0 = top_data + (p) * top_cstep;
					int* outptr1 = top_data + (p + 1) * top_cstep;
					int* outptr2 = top_data + (p + 2) * top_cstep;
					int* outptr3 = top_data + (p + 3) * top_cstep;

					int i = 0;

					for (; i + 7 < size; i += 8)
					{
						const signed char* tmpptr = tmp_data + (i / 8) * tmp_cstep;
						const signed char* kptr = kernel_tm_int8_data + (p / 4) * kernel_tm_int8_cstep;

#if __ARM_NEON
						asm volatile(
							// inch loop
							"vmov.s32    q6, #0            \n"
							"vmov.s32    q7, #0            \n"
							"vmov.s32    q8, #0            \n"
							"vmov.s32    q9, #0            \n"
							"vmov.s32    q10, #0           \n"
							"vmov.s32    q11, #0           \n"
							"vmov.s32    q12, #0           \n"
							"vmov.s32    q13, #0           \n"

							"lsr         r4, %12, #2       \n" // r4 = nn = inch >> 2
							"cmp         r4, #0            \n"
							"beq         1f                \n"

							"0:                            \n" // for(; nn != 0; nn--)
							"pld         [%4, #128]        \n"
							"vld1.s8     {d4-d7}, [%4]!    \n" // tmpr a00-a07,a10-a17,a20-a27,a30-a37    a(inch)(data)
							"vmovl.s8    q5, d7            \n" // a30-a37
							"vmovl.s8    q4, d6            \n" // a20-a27
							"vmovl.s8    q3, d5            \n" // a10-a17
							"vmovl.s8    q2, d4            \n" // a00-a07

							"vld1.s8     {d0-d1}, [%5]!    \n" // kptr k00-k30,k01-k31,k02-k32,k03-k33    k(outch)(inch)
							"vmovl.s8    q1, d1            \n" // k02-k32,k03-k33
							"vmovl.s8    q0, d0            \n" // k00-k30,k01-k31

							"vmlal.s16   q6, d4, d0[0]     \n" // sum0 = (a00-a07) * k00
							"vmlal.s16   q7, d5, d0[0]     \n"
							"vmlal.s16   q8, d4, d0[1]     \n" // sum1 = (a00-a07) * k10
							"vmlal.s16   q9, d5, d0[1]     \n"
							"vmlal.s16   q10, d4, d0[2]    \n" // sum2 = (a00-a07) * k20
							"vmlal.s16   q11, d5, d0[2]    \n"
							"vmlal.s16   q12, d4, d0[3]    \n" // sum3 = (a00-a07) * k30
							"vmlal.s16   q13, d5, d0[3]    \n"

							"vmlal.s16   q6, d6, d1[0]     \n" // sum0 += (a10-a17) * k01
							"vmlal.s16   q7, d7, d1[0]     \n"
							"vmlal.s16   q8, d6, d1[1]     \n" // sum1 += (a10-a17) * k11
							"vmlal.s16   q9, d7, d1[1]     \n"
							"vmlal.s16   q10, d6, d1[2]    \n" // sum2 += (a10-a17) * k21
							"vmlal.s16   q11, d7, d1[2]    \n"
							"vmlal.s16   q12, d6, d1[3]    \n" // sum3 += (a10-a17) * k31
							"vmlal.s16   q13, d7, d1[3]    \n"

							"vmlal.s16   q6, d8, d2[0]     \n" // sum0 += (a20-a27) * k02
							"vmlal.s16   q7, d9, d2[0]     \n"
							"vmlal.s16   q8, d8, d2[1]     \n" // sum1 += (a20-a27) * k12
							"vmlal.s16   q9, d9, d2[1]     \n"
							"vmlal.s16   q10, d8, d2[2]    \n" // sum2 += (a20-a27) * k22
							"vmlal.s16   q11, d9, d2[2]    \n"
							"vmlal.s16   q12, d8, d2[3]    \n" // sum3 += (a20-a27) * k32
							"vmlal.s16   q13, d9, d2[3]    \n"

							"vmlal.s16   q6, d10, d3[0]    \n" // sum0 += (a30-a37) * k03
							"vmlal.s16   q7, d11, d3[0]    \n"
							"vmlal.s16   q8, d10, d3[1]    \n" // sum1 += (a30-a37) * k13
							"vmlal.s16   q9, d11, d3[1]    \n"
							"vmlal.s16   q10, d10, d3[2]   \n" // sum2 += (a30-a37) * k23
							"vmlal.s16   q11, d11, d3[2]   \n"
							"vmlal.s16   q12, d10, d3[3]   \n" // sum3 += (a30-a37) * k33
							"vmlal.s16   q13, d11, d3[3]   \n"

							"subs        r4, r4, #1        \n"
							"bne         0b                \n" // end for

							"1:                            \n"
							// remain loop
							"and         r4, %12, #3       \n" // r4 = remain = inch & 3
							"cmp         r4, #0            \n"
							"beq         3f                \n"

							"2:                            \n" // for(; remain != 0; remain--)
							"vld1.s8     {d2}, [%4]!       \n" // tmpr a00-a07    a(inch)(data)
							"vld1.s8     {d0}, [%5]        \n" // kptr k00-k30    k(outch)(inch)
							"vmovl.s8    q1, d2            \n"
							"vmovl.s8    q0, d0            \n"
							"add         %5, #4            \n"

							"vmlal.s16   q6, d2, d0[0]     \n" // sum0 += (a00-a07) * k00
							"vmlal.s16   q7, d3, d0[0]     \n"
							"vmlal.s16   q8, d2, d0[1]     \n" // sum1 += (a00-a07) * k10
							"vmlal.s16   q9, d3, d0[1]     \n"
							"vmlal.s16   q10, d2, d0[2]    \n" // sum2 += (a00-a07) * k20
							"vmlal.s16   q11, d3, d0[2]    \n"
							"vmlal.s16   q12, d2, d0[3]    \n" // sum3 += (a00-a07) * k30
							"vmlal.s16   q13, d3, d0[3]    \n"

							"subs        r4, r4, #1        \n"
							"bne         2b                \n"

							"3:                            \n" // store the result to memory
							"vst1.s32    {d12-d15}, [%0]!  \n"
							"vst1.s32    {d16-d19}, [%1]!  \n"
							"vst1.s32    {d20-d23}, [%2]!  \n"
							"vst1.s32    {d24-d27}, [%3]!  \n"

							: "=r"(outptr0), // %0
							"=r"(outptr1), // %1
							"=r"(outptr2), // %2
							"=r"(outptr3), // %3
							"=r"(tmpptr),  // %4
							"=r"(kptr)     // %5
							: "0"(outptr0),
							"1"(outptr1),
							"2"(outptr2),
							"3"(outptr3),
							"4"(tmpptr),
							"5"(kptr),
							"r"(inch) // %12
							: "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#else
						int sum0_0 = 0;
						int sum0_1 = 0;
						int sum0_2 = 0;
						int sum0_3 = 0;
						int sum0_4 = 0;
						int sum0_5 = 0;
						int sum0_6 = 0;
						int sum0_7 = 0;

						int sum1_0 = 0;
						int sum1_1 = 0;
						int sum1_2 = 0;
						int sum1_3 = 0;
						int sum1_4 = 0;
						int sum1_5 = 0;
						int sum1_6 = 0;
						int sum1_7 = 0;

						int sum2_0 = 0;
						int sum2_1 = 0;
						int sum2_2 = 0;
						int sum2_3 = 0;
						int sum2_4 = 0;
						int sum2_5 = 0;
						int sum2_6 = 0;
						int sum2_7 = 0;

						int sum3_0 = 0;
						int sum3_1 = 0;
						int sum3_2 = 0;
						int sum3_3 = 0;
						int sum3_4 = 0;
						int sum3_5 = 0;
						int sum3_6 = 0;
						int sum3_7 = 0;

						for (int q = 0; q < inch; q++)
						{
							sum0_0 += tmpptr[0] * kptr[0];
							sum0_1 += tmpptr[1] * kptr[0];
							sum0_2 += tmpptr[2] * kptr[0];
							sum0_3 += tmpptr[3] * kptr[0];
							sum0_4 += tmpptr[4] * kptr[0];
							sum0_5 += tmpptr[5] * kptr[0];
							sum0_6 += tmpptr[6] * kptr[0];
							sum0_7 += tmpptr[7] * kptr[0];

							sum1_0 += tmpptr[0] * kptr[1];
							sum1_1 += tmpptr[1] * kptr[1];
							sum1_2 += tmpptr[2] * kptr[1];
							sum1_3 += tmpptr[3] * kptr[1];
							sum1_4 += tmpptr[4] * kptr[1];
							sum1_5 += tmpptr[5] * kptr[1];
							sum1_6 += tmpptr[6] * kptr[1];
							sum1_7 += tmpptr[7] * kptr[1];

							sum2_0 += tmpptr[0] * kptr[2];
							sum2_1 += tmpptr[1] * kptr[2];
							sum2_2 += tmpptr[2] * kptr[2];
							sum2_3 += tmpptr[3] * kptr[2];
							sum2_4 += tmpptr[4] * kptr[2];
							sum2_5 += tmpptr[5] * kptr[2];
							sum2_6 += tmpptr[6] * kptr[2];
							sum2_7 += tmpptr[7] * kptr[2];

							sum3_0 += tmpptr[0] * kptr[3];
							sum3_1 += tmpptr[1] * kptr[3];
							sum3_2 += tmpptr[2] * kptr[3];
							sum3_3 += tmpptr[3] * kptr[3];
							sum3_4 += tmpptr[4] * kptr[3];
							sum3_5 += tmpptr[5] * kptr[3];
							sum3_6 += tmpptr[6] * kptr[3];
							sum3_7 += tmpptr[7] * kptr[3];

							tmpptr += 8;
							kptr += 4;
						}

						outptr0[0] = sum0_0;
						outptr0[1] = sum0_1;
						outptr0[2] = sum0_2;
						outptr0[3] = sum0_3;
						outptr0[4] = sum0_4;
						outptr0[5] = sum0_5;
						outptr0[6] = sum0_6;
						outptr0[7] = sum0_7;

						outptr1[0] = sum1_0;
						outptr1[1] = sum1_1;
						outptr1[2] = sum1_2;
						outptr1[3] = sum1_3;
						outptr1[4] = sum1_4;
						outptr1[5] = sum1_5;
						outptr1[6] = sum1_6;
						outptr1[7] = sum1_7;

						outptr2[0] = sum2_0;
						outptr2[1] = sum2_1;
						outptr2[2] = sum2_2;
						outptr2[3] = sum2_3;
						outptr2[4] = sum2_4;
						outptr2[5] = sum2_5;
						outptr2[6] = sum2_6;
						outptr2[7] = sum2_7;

						outptr3[0] = sum3_0;
						outptr3[1] = sum3_1;
						outptr3[2] = sum3_2;
						outptr3[3] = sum3_3;
						outptr3[4] = sum3_4;
						outptr3[5] = sum3_5;
						outptr3[6] = sum3_6;
						outptr3[7] = sum3_7;

						outptr0 += 8;
						outptr1 += 8;
						outptr2 += 8;
						outptr3 += 8;
#endif // __ARM_NEON
					}

					for (; i + 3 < size; i += 4)
					{
						const signed char* tmpptr = tmp_data + (i / 8 + (i % 8) / 4) * tmp_cstep;
						const signed char* kptr = kernel_tm_int8_data + (p / 4) * kernel_tm_int8_cstep;

#if __ARM_NEON
						asm volatile(
							// inch loop
							"vmov.s32    q6, #0            \n"
							"vmov.s32    q7, #0            \n"
							"vmov.s32    q8, #0            \n"
							"vmov.s32    q9, #0            \n"

							"lsr         r4, %12, #2       \n" // r4 = nn = inch >> 2
							"cmp         r4, #0            \n"
							"beq         1f                \n"

							"0:                            \n" // for(; nn != 0; nn--)
							"pld         [%4, #128]        \n"
							"vld1.s8     {d4-d5}, [%4]!    \n" // tmpr a00-a03,a10-a13,a20-a23,a30-a33    a(inch)(data)
							"vmovl.s8    q3, d5            \n" // a20-a23,a30-a33
							"vmovl.s8    q2, d4            \n" // a00-a04,a10-a14

							"vld1.s8     {d0-d1}, [%5]!    \n" // kptr k00-k30,k01-k31,k02-k32,k03-k33    k(outch)(inch)
							"vmovl.s8    q1, d1            \n" // k02-k32,k03-k33
							"vmovl.s8    q0, d0            \n" // k00-k30,k01-k31

							"vmlal.s16   q6, d4, d0[0]     \n" // sum0 = (a00-a03) * k00
							"vmlal.s16   q7, d4, d0[1]     \n" // sum1 = (a00-a03) * k10
							"vmlal.s16   q8, d4, d0[2]     \n" // sum2 = (a00-a03) * k20
							"vmlal.s16   q9, d4, d0[3]     \n" // sum3 = (a00-a03) * k30

							"vmlal.s16   q6, d5, d1[0]     \n" // sum0 += (a10-a13) * k01
							"vmlal.s16   q7, d5, d1[1]     \n" // sum1 += (a10-a13) * k11
							"vmlal.s16   q8, d5, d1[2]     \n" // sum2 += (a10-a13) * k21
							"vmlal.s16   q9, d5, d1[3]     \n" // sum3 += (a10-a13) * k31

							"vmlal.s16   q6, d6, d2[0]     \n" // sum0 += (a20-a23) * k02
							"vmlal.s16   q7, d6, d2[1]     \n" // sum1 += (a20-a23) * k12
							"vmlal.s16   q8, d6, d2[2]     \n" // sum2 += (a20-a23) * k22
							"vmlal.s16   q9, d6, d2[3]     \n" // sum3 += (a20-a23) * k32

							"vmlal.s16   q6, d7, d3[0]     \n" // sum0 += (a30-a33) * k03
							"vmlal.s16   q7, d7, d3[1]     \n" // sum1 += (a30-a33) * k13
							"vmlal.s16   q8, d7, d3[2]     \n" // sum2 += (a30-a33) * k23
							"vmlal.s16   q9, d7, d3[3]     \n" // sum3 += (a30-a33) * k33

							"subs        r4, r4, #1        \n"
							"bne         0b                \n" // end for

							"1:                            \n"
							// remain loop
							"and         r4, %12, #3       \n" // r4 = remain = inch & 3
							"cmp         r4, #0            \n"
							"beq         3f                \n"

							"2:                            \n" // for(; remain != 0; remain--)
							"vld1.s8     {d2}, [%4]        \n" // tmpr a00-a03    a(inch)(data)
							"vld1.s8     {d0}, [%5]        \n" // kptr k00-k30    k(outch)(inch)
							"vmovl.s8    q1, d2            \n"
							"vmovl.s8    q0, d0            \n"
							"add         %4, #4            \n"
							"add         %5, #4            \n"

							"vmlal.s16   q6, d2, d0[0]     \n" // sum0 += (a00-a03) * k00
							"vmlal.s16   q7, d2, d0[1]     \n" // sum1 += (a00-a03) * k10
							"vmlal.s16   q8, d2, d0[2]     \n" // sum2 += (a00-a03) * k20
							"vmlal.s16   q9, d2, d0[3]     \n" // sum3 += (a00-a03) * k30

							"subs        r4, r4, #1        \n"
							"bne         2b                \n"

							"3:                            \n" // store the result to memory
							"vst1.s32    {d12-d13}, [%0]!  \n"
							"vst1.s32    {d14-d15}, [%1]!  \n"
							"vst1.s32    {d16-d17}, [%2]!  \n"
							"vst1.s32    {d18-d19}, [%3]!  \n"

							: "=r"(outptr0), // %0
							"=r"(outptr1), // %1
							"=r"(outptr2), // %2
							"=r"(outptr3), // %3
							"=r"(tmpptr),  // %4
							"=r"(kptr)     // %5
							: "0"(outptr0),
							"1"(outptr1),
							"2"(outptr2),
							"3"(outptr3),
							"4"(tmpptr),
							"5"(kptr),
							"r"(inch) // %12
							: "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#else
						int sum0_0 = 0;
						int sum0_1 = 0;
						int sum0_2 = 0;
						int sum0_3 = 0;

						int sum1_0 = 0;
						int sum1_1 = 0;
						int sum1_2 = 0;
						int sum1_3 = 0;

						int sum2_0 = 0;
						int sum2_1 = 0;
						int sum2_2 = 0;
						int sum2_3 = 0;

						int sum3_0 = 0;
						int sum3_1 = 0;
						int sum3_2 = 0;
						int sum3_3 = 0;

						for (int q = 0; q < inch; q++)
						{
							sum0_0 += tmpptr[0] * kptr[0];
							sum0_1 += tmpptr[1] * kptr[0];
							sum0_2 += tmpptr[2] * kptr[0];
							sum0_3 += tmpptr[3] * kptr[0];

							sum1_0 += tmpptr[0] * kptr[1];
							sum1_1 += tmpptr[1] * kptr[1];
							sum1_2 += tmpptr[2] * kptr[1];
							sum1_3 += tmpptr[3] * kptr[1];

							sum2_0 += tmpptr[0] * kptr[2];
							sum2_1 += tmpptr[1] * kptr[2];
							sum2_2 += tmpptr[2] * kptr[2];
							sum2_3 += tmpptr[3] * kptr[2];

							sum3_0 += tmpptr[0] * kptr[3];
							sum3_1 += tmpptr[1] * kptr[3];
							sum3_2 += tmpptr[2] * kptr[3];
							sum3_3 += tmpptr[3] * kptr[3];

							tmpptr += 4;
							kptr += 4;
						}

						outptr0[0] = sum0_0;
						outptr0[1] = sum0_1;
						outptr0[2] = sum0_2;
						outptr0[3] = sum0_3;

						outptr1[0] = sum1_0;
						outptr1[1] = sum1_1;
						outptr1[2] = sum1_2;
						outptr1[3] = sum1_3;

						outptr2[0] = sum2_0;
						outptr2[1] = sum2_1;
						outptr2[2] = sum2_2;
						outptr2[3] = sum2_3;

						outptr3[0] = sum3_0;
						outptr3[1] = sum3_1;
						outptr3[2] = sum3_2;
						outptr3[3] = sum3_3;

						outptr0 += 4;
						outptr1 += 4;
						outptr2 += 4;
						outptr3 += 4;
#endif // __ARM_NEON
					}

					for (; i < size; i++)
					{
						const signed char* tmpptr = tmp_data + (i / 8 + (i % 8) / 4 + i % 4) * tmp_cstep;
						const signed char* kptr = kernel_tm_int8_data + (p / 4) * kernel_tm_int8_cstep;

#if __ARM_NEON
						asm volatile(
							// inch loop
							"veor        q6, q6, q6        \n"
							"veor        q7, q7, q7        \n"
							"veor        q8, q8, q8        \n"
							"veor        q9, q9, q9        \n"
							"vmov.s32    q10, #0           \n"

							"lsr         r4, %12, #2       \n" // r4 = nn = inch >> 2
							"cmp         r4, #0            \n"
							"beq         1f                \n"

							"0:                            \n" // for(; nn != 0; nn--)
							"pld         [%4, #128]        \n"
							"vld1.s8     {d4}, [%4]        \n" // tmpr a00,a10,a20,a30    a(inch)(data)
							"add         %4, #4            \n"
							"vmovl.s8    q2, d4            \n" // a00,a10,a20,a30

							"vld1.s8     {d0-d1}, [%5]!    \n" // kptr k00-k30,k01-k31,k02-k32,k03-k33    k(outch)(inch)
							"vmovl.s8    q1, d1            \n" // k02-k32,k03-k33
							"vmovl.s8    q0, d0            \n" // k00-k30,k01-k31

							"vmlal.s16   q6, d0, d4[0]     \n" // (k00-k30) * a00
							"vmlal.s16   q7, d1, d4[1]     \n" // (k01-k31) * a10
							"vmlal.s16   q8, d2, d4[2]     \n" // (k02-k32) * a20
							"vmlal.s16   q9, d3, d4[3]     \n" // (k03-k33) * a30

							"subs        r4, r4, #1        \n"
							"bne         0b                \n" // end for

							"vadd.s32    q6, q6, q7        \n"
							"vadd.s32    q9, q9, q8        \n"
							"vadd.s32    q10, q6, q9       \n"

							"1:                            \n"
							// remain loop
							"and         r4, %12, #3       \n" // r4 = remain = inch & 3
							"cmp         r4, #0            \n"
							"beq         3f                \n"

							"2:                            \n" // for(; remain != 0; remain--)
							"vld1.s8     {d2}, [%4]        \n" // tmpr a00        a(inch)(data)
							"vld1.s8     {d0}, [%5]        \n" // kptr k00-k30    k(outch)(inch)
							"vmovl.s8    q1, d2            \n"
							"vmovl.s8    q0, d0            \n"
							"add         %4, #1            \n"
							"add         %5, #4            \n"

							"vmlal.s16   q10, d0, d2[0]    \n"

							"subs        r4, r4, #1        \n"
							"bne         2b                \n"

							"3:                            \n" // store the result to memory
							"vst1.s32    {d20[0]}, [%0]!   \n"
							"vst1.s32    {d20[1]}, [%1]!   \n"
							"vst1.s32    {d21[0]}, [%2]!   \n"
							"vst1.s32    {d21[1]}, [%3]!   \n"

							: "=r"(outptr0), // %0
							"=r"(outptr1), // %1
							"=r"(outptr2), // %2
							"=r"(outptr3), // %3
							"=r"(tmpptr),  // %4
							"=r"(kptr)     // %5
							: "0"(outptr0),
							"1"(outptr1),
							"2"(outptr2),
							"3"(outptr3),
							"4"(tmpptr),
							"5"(kptr),
							"r"(inch) // %12
							: "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#else
						int sum0 = 0;
						int sum1 = 0;
						int sum2 = 0;
						int sum3 = 0;

						for (int q = 0; q < inch; q++)
						{
							sum0 += tmpptr[0] * kptr[0];
							sum1 += tmpptr[0] * kptr[1];
							sum2 += tmpptr[0] * kptr[2];
							sum3 += tmpptr[0] * kptr[3];

							tmpptr++;
							kptr += 4;
						}

						outptr0[0] = sum0;
						outptr1[0] = sum1;
						outptr2[0] = sum2;
						outptr3[0] = sum3;

						outptr0++;
						outptr1++;
						outptr2++;
						outptr3++;
#endif // __ARM_NEON
				}
		}

				remain_outch_start += nn_outch << 2;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
				for (int p = remain_outch_start; p < outch; p++)
				{
					int* out0 = top_data + (p) * top_cstep;

					int* outptr0 = out0;

					int i = 0;

					for (; i + 7 < size; i += 8)
					{
						const signed char* tmpptr = tmp_data + (i / 8) * tmp_cstep;
						const signed char* kptr = kernel_tm_int8_data + (p / 4 + p % 4) * kernel_tm_int8_cstep;

#if __ARM_NEON
						asm volatile(
							// inch loop
							"vmov.s32    q6, #0            \n"
							"vmov.s32    q7, #0            \n"

							"lsr         r4, %6, #2        \n" // r4 = nn = inch >> 2
							"cmp         r4, #0            \n"
							"beq         1f                \n"

							"0:                            \n" // for(; nn != 0; nn--)
							"pld         [%1, #128]        \n"
							"vld1.s8     {d4-d7}, [%1]!    \n" // tmpr a00-a07,a10-a17,a20-a27,a30-a37    a(inch)(data)
							"vmovl.s8    q5, d7            \n" // a30-a37
							"vmovl.s8    q4, d6            \n" // a20-a27
							"vmovl.s8    q3, d5            \n" // a10-a17
							"vmovl.s8    q2, d4            \n" // a00-a07

							"vld1.s8     {d0}, [%2]        \n" // kptr k00,k01,k02,k03    k(outch)(inch)
							"vmovl.s8    q0, d0            \n" // k00,k01,k02,k03
							"add         %2, #4            \n"

							"vmlal.s16   q6, d4, d0[0]     \n" // (a00-a07) * k00
							"vmlal.s16   q7, d5, d0[0]     \n"
							"vmlal.s16   q6, d6, d0[1]     \n" // (a10-a17) * k01
							"vmlal.s16   q7, d7, d0[1]     \n"
							"vmlal.s16   q6, d8, d0[2]     \n" // (a20-a27) * k02
							"vmlal.s16   q7, d9, d0[2]     \n"
							"vmlal.s16   q6, d10, d0[3]    \n" // (a30-a37) * k03
							"vmlal.s16   q7, d11, d0[3]    \n"

							"subs        r4, r4, #1        \n"
							"bne         0b                \n" // end for

							"1:                            \n"
							// remain loop
							"and         r4, %6, #3        \n" // r4 = remain = inch & 3
							"cmp         r4, #0            \n"
							"beq         3f                \n"

							"2:                            \n" // for(; remain != 0; remain--)
							"vld1.s8     {d2}, [%1]!       \n" // tmpr a00-a07    a(inch)(data)
							"vld1.s8     {d0}, [%2]        \n" // kptr k00        k(outch)(inch)
							"vmovl.s8    q1, d2            \n"
							"vmovl.s8    q0, d0            \n"
							"add         %2, #1            \n"

							"vmlal.s16   q6, d2, d0[0]     \n" // (a00-a07) * k00
							"vmlal.s16   q7, d3, d0[0]     \n"

							"subs        r4, r4, #1        \n"
							"bne         2b                \n"

							"3:                            \n" // store the result to memory
							"vst1.s32    {d12-d15}, [%0]!  \n"

							: "=r"(outptr0), // %0
							"=r"(tmpptr),  // %1
							"=r"(kptr)     // %2
							: "0"(outptr0),
							"1"(tmpptr),
							"2"(kptr),
							"r"(inch) // %6
							: "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
#else
						int sum0 = 0;
						int sum1 = 0;
						int sum2 = 0;
						int sum3 = 0;
						int sum4 = 0;
						int sum5 = 0;
						int sum6 = 0;
						int sum7 = 0;

						for (int q = 0; q < inch; q++)
						{
							sum0 += tmpptr[0] * kptr[0];
							sum1 += tmpptr[1] * kptr[0];
							sum2 += tmpptr[2] * kptr[0];
							sum3 += tmpptr[3] * kptr[0];
							sum4 += tmpptr[4] * kptr[0];
							sum5 += tmpptr[5] * kptr[0];
							sum6 += tmpptr[6] * kptr[0];
							sum7 += tmpptr[7] * kptr[0];

							tmpptr += 8;
							kptr++;
						}

						outptr0[0] = sum0;
						outptr0[1] = sum1;
						outptr0[2] = sum2;
						outptr0[3] = sum3;
						outptr0[4] = sum4;
						outptr0[5] = sum5;
						outptr0[6] = sum6;
						outptr0[7] = sum7;

						outptr0 += 8;
#endif // __ARM_NEON
					}

					for (; i + 3 < size; i += 4)
					{
						const signed char* tmpptr = tmp_data + (i / 8 + (i % 8) / 4) * tmp_cstep;
						const signed char* kptr = kernel_tm_int8_data + (p / 4 + p % 4) * kernel_tm_int8_cstep;

#if __ARM_NEON
						asm volatile(
							// inch loop
							"vmov.s32    q6, #0            \n"

							"lsr         r4, %6, #2        \n" // r4 = nn = inch >> 2
							"cmp         r4, #0            \n"
							"beq         1f                \n"

							"0:                            \n" // for(; nn != 0; nn--)
							"pld         [%2, #128]        \n"
							"vld1.s8     {d4-d5}, [%1]!    \n" // tmpr a00-a03,a10-a13,a20-a23,a30-a33    a(inch)(data)
							"vmovl.s8    q3, d5            \n" // a20-a23,a30-a33
							"vmovl.s8    q2, d4            \n" // a00-a03,a10-a13

							"vld1.s8     {d0}, [%2]        \n" // kptr k00,k01,k02,k03    k(outch)(inch)
							"vmovl.s8    q0, d0            \n" // k00,k01,k02,k03
							"add         %2, #4            \n"

							"vmlal.s16   q6, d4, d0[0]     \n" // (a00-a03) * k00
							"vmlal.s16   q6, d5, d0[1]     \n" // (a10-a13) * k01
							"vmlal.s16   q6, d6, d0[2]     \n" // (a20-a23) * k02
							"vmlal.s16   q6, d7, d0[3]     \n" // (a30-a33) * k03

							"subs        r4, r4, #1        \n"
							"bne         0b                \n" // end for

							"1:                            \n"
							// remain loop
							"and         r4, %6, #3        \n" // r4 = remain = inch & 3
							"cmp         r4, #0            \n"
							"beq         3f                \n"

							"2:                            \n" // for(; remain != 0; remain--)
							"vld1.s8     {d2}, [%1]        \n" // tmpr a00-a03    a(inch)(data)
							"vld1.s8     {d0}, [%2]        \n" // kptr k00        k(outch)(inch)
							"vmovl.s8    q1, d2            \n"
							"vmovl.s8    q0, d0            \n"
							"add         %1, #4            \n"
							"add         %2, #1            \n"

							"vmlal.s16   q6, d2, d0[0]     \n" // (a00-a03) * k00

							"subs        r4, r4, #1        \n"
							"bne         2b                \n"

							"3:                            \n" // store the result to memory
							"vst1.s32    {d12-d13}, [%0]!  \n"

							: "=r"(outptr0), // %0
							"=r"(tmpptr),  // %1
							"=r"(kptr)     // %2
							: "0"(outptr0),
							"1"(tmpptr),
							"2"(kptr),
							"r"(inch) // %6
							: "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6");
#else
						int sum0 = 0;
						int sum1 = 0;
						int sum2 = 0;
						int sum3 = 0;

						for (int q = 0; q < inch; q++)
						{
							sum0 += tmpptr[0] * kptr[0];
							sum1 += tmpptr[1] * kptr[0];
							sum2 += tmpptr[2] * kptr[0];
							sum3 += tmpptr[3] * kptr[0];

							tmpptr += 4;
							kptr++;
						}

						outptr0[0] = sum0;
						outptr0[1] = sum1;
						outptr0[2] = sum2;
						outptr0[3] = sum3;

						outptr0 += 4;
#endif // __ARM_NEON
				}

					for (; i < size; i++)
					{
						const signed char* tmpptr = tmp_data + (i / 8 + (i % 8) / 4 + i % 4) * tmp_cstep;
						const signed char* kptr = kernel_tm_int8_data + (p / 4 + p % 4)* kernel_tm_int8_cstep;

						int q = 0;
						int sum0 = 0;

						for (; q < inch; q++)
						{
							sum0 += tmpptr[0] * kptr[0];
							tmpptr++;
							kptr++;
						}

						outptr0[0] = sum0;

						outptr0++;
					}
				}

				//     // NOTE sgemm int8
				//     for (; p<outch; p++)
				//     {
				//         Mat out0 = top_blob.channel(p);
				//
				//         int* outptr0 = out0;
				//
				//         for (int i=0; i<size; i++)
				//         {
				//             int sum = 0;
				//
				//             const signed char* kptr = _kernel.channel(p/8 + p%8);
				//
				//             for (int q=0; q<inch; q++)
				//             {
				//                 const signed char* img0 = bottom_blob.channel(q);
				//
				//                 sum += img0[i] * kptr[0];
				//                 kptr ++;
				//             }
				//
				//             outptr0[i] = sum;
				//         }
				//     }
			}
		}
#endif

#if __aarch64__
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

			const int8_t* kernel_tm_int8_data = kernel_tm_int8_->cpu_data();

			int top_cstep = outw * outh;
			int kernel_size = this->kernel_size_h_ * this->kernel_size_w_;
			int out_size = outw * outh;

			// im2col
			memory::tensor<int8_t> bottom_im2col(std::vector<int>{1, 1, kernel_size* inch, out_size}, this->params_.device_, memory::NCHW);
			int8_t* ret = bottom_im2col.mutable_cpu_data();
			const int8_t* bottom_im2col_data = bottom_im2col.cpu_data();

			memory::tensor<int8_t> bottom_tm(out_size * inch * kernel_size, this->params_.device_, memory::NCHW);
			int8_t* bottom_tm_data = bottom_tm.mutable_cpu_data();

			for (size_t num_i = 0; num_i < num; num_i++)
			{
				const int8_t* bottom_data = bottom->cpu_data() + num_i * inch * bottom_cstep;
				int* top_data = top->mutable_cpu_data() + num_i * outch * top_cstep;

				{
					const int stride = kernel_size * out_size;
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int p = 0; p < inch; p++)
					{
						const signed char* input = bottom_data + (p)*bottom_cstep;
						int retID = stride * p;
						for (int u = 0; u < this->kernel_size_h_; u++)
						{
							for (int v = 0; v < this->kernel_size_w_; v++)
							{
								for (int i = 0; i < outh; i++)
								{
									for (int j = 0; j < outw; j++)
									{
										int row = u + i * this->stride_h_;
										int col = v + j * this->stride_w_;
										int index = row * w + col;
										ret[retID] = input[index];
										retID++;
									}
								}
							}
						}
					}
				}

				const int m = outch;
				const int n = out_size;
				const int k = inch * kernel_size;

				{
					const int8_t* pData = bottom_im2col_data;
					int8_t* pReorder = bottom_tm_data;
					reorder_b(pData, pReorder, k, n, n);
				}
				// GEMM
				int* pc = top_data;
				const int8_t* pa = kernel_tm_int8_data;
				int8_t* pb = bottom_tm_data;
				const int ldc = top_cstep;

				int8kernel((void*)pc, pa, pb, m, k, n, ldc, 0, 0);
			}
		}
#else
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

			const int8_t* kernel_tm_int8_data = kernel_tm_int8_->cpu_data();
			int kernel_tm_int8_cstep = kernel_tm_int8_->count(2, 4);

			int top_cstep = outw * outh;
			int kernel_size = this->kernel_size_h_ * this->kernel_size_w_;
			int out_size = outw * outh;

			// im2col
			memory::tensor<int8_t> bottom_im2col(std::vector<int>{1, 1, kernel_size* inch, out_size}, this->params_.device_, memory::NCHW);
			int8_t* ret = bottom_im2col.mutable_cpu_data();
			const int8_t* bottom_im2col_data = bottom_im2col.cpu_data();

			// bottom_im2col memory packed 8 x 8
			memory::tensor<int8_t> bottom_tm(std::vector<int>{1, out_size / 8 + out_size % 8, inch, 8 * kernel_size}, this->params_.device_, memory::NCHW);
			int8_t* bottom_tm_data = bottom_tm.mutable_cpu_data();
			int bottom_tm_cstep = bottom_tm.count(2, 4);

			for (int num_i = 0; num_i < num; num_i++)
			{
				const int8_t* bottom_data = bottom->cpu_data() + num_i * inch * bottom_cstep;
				int* top_data = top->mutable_cpu_data() + num_i * outch * top_cstep;

				{
					const int stride = kernel_size * out_size;
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int p = 0; p < inch; p++)
					{
						const signed char* input = bottom_data + (p) * bottom_cstep;
						int retID = stride * p;
						for (int u = 0; u < this->kernel_size_h_; u++)
						{
							for (int v = 0; v < this->kernel_size_w_; v++)
							{
								for (int i = 0; i < outh; i++)
								{
									for (int j = 0; j < outw; j++)
									{
										int row = u + i * this->stride_h_;
										int col = v + j * this->stride_w_;
										int index = row * w + col;
										ret[retID] = input[index];
										retID++;
									}
								}
							}
						}
					}
				}

				{
					int nn_size = out_size >> 3;
					int remain_size_start = nn_size << 3;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int ii = 0; ii < nn_size; ii++)
					{
						int i = ii * 8;

						const signed char* img0 = bottom_im2col_data;
						img0 += i;

						signed char* tmpptr = bottom_tm_data + (i / 8) * bottom_tm_cstep;

						for (int q = 0; q < inch * kernel_size; q++)
						{
#if __ARM_NEON
#if __aarch64__
							asm volatile(
								"prfm    pldl1keep, [%0, #64]    \n"
								"ld1     {v0.8b}, [%0]           \n"
								"st1     {v0.8b}, [%1]           \n"
								: "=r"(img0),  // %0
								"=r"(tmpptr) // %1
								: "0"(img0),
								"1"(tmpptr)
								: "cc", "memory", "v0");
#else
							asm volatile(
								"pld        [%0, #64]     \n"
								"vld1.s8   {d0}, [%0]     \n"
								"vst1.s8   {d0}, [%1]     \n"
								: "=r"(img0),  // %0
								"=r"(tmpptr) // %1
								: "0"(img0),
								"1"(tmpptr)
								: "cc", "memory", "d0");
#endif // __aarch64__
#else
							tmpptr[0] = img0[0];
							tmpptr[1] = img0[1];
							tmpptr[2] = img0[2];
							tmpptr[3] = img0[3];
							tmpptr[4] = img0[4];
							tmpptr[5] = img0[5];
							tmpptr[6] = img0[6];
							tmpptr[7] = img0[7];
#endif // __ARM_NEON
							tmpptr += 8;
							img0 += out_size;
						}
					}

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int i = remain_size_start; i < out_size; i++)
					{
						const signed char* img0 = bottom_im2col_data;
						img0 += i;

						signed char* tmpptr = bottom_tm_data + (i / 8 + i % 8) * bottom_tm_cstep;

						for (int q = 0; q < inch * kernel_size; q++)
						{
							tmpptr[0] = img0[0];

							tmpptr += 1;
							img0 += out_size;
						}
					}
				}

				// sgemm(int M, int N, int L, float* A, float* B, float* C)
				{
					//int M = outch;  // outch
					int N = out_size;                // outsize or out stride
					int L = kernel_size * inch; // ksize * inch

					int nn_outch = 0;
					int remain_outch_start = 0;

#if __ARM_NEON && __aarch64__
					nn_outch = outch >> 3;
					remain_outch_start = nn_outch << 3;
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int pp = 0; pp < nn_outch; pp++)
					{
						int i = pp * 8;

						int* output0 = top_data + (i) * top_cstep;
						int* output1 = top_data + (i + 1) * top_cstep;
						int* output2 = top_data + (i + 2) * top_cstep;
						int* output3 = top_data + (i + 3) * top_cstep;
						int* output4 = top_data + (i + 4) * top_cstep;
						int* output5 = top_data + (i + 5) * top_cstep;
						int* output6 = top_data + (i + 6) * top_cstep;
						int* output7 = top_data + (i + 7) * top_cstep;

						int j = 0;
						for (; j + 7 < N; j = j + 8)
						{
							signed char* vb = bottom_tm_data + (j / 8) * bottom_tm_cstep;
							const signed char* va = kernel_tm_int8_data + (i / 8) * kernel_tm_int8_cstep;
#if __aarch64__
							asm volatile(
								"eor    v16.16b, v16.16b, v16.16b    \n" // sum0
								"eor    v17.16b, v17.16b, v17.16b    \n" // sum0n
								"eor    v18.16b, v18.16b, v18.16b    \n" // sum1
								"eor    v19.16b, v19.16b, v19.16b    \n" // sum1n
								"eor    v20.16b, v20.16b, v20.16b    \n" // sum2
								"eor    v21.16b, v21.16b, v21.16b    \n" // sum2n
								"eor    v22.16b, v22.16b, v22.16b    \n" // sum3
								"eor    v23.16b, v23.16b, v23.16b    \n" // sum3n
								"eor    v24.16b, v24.16b, v24.16b    \n" // sum4
								"eor    v25.16b, v25.16b, v25.16b    \n" // sum4n
								"eor    v26.16b, v26.16b, v26.16b    \n" // sum5
								"eor    v27.16b, v27.16b, v27.16b    \n" // sum5n
								"eor    v28.16b, v28.16b, v28.16b    \n" // sum6
								"eor    v29.16b, v29.16b, v29.16b    \n" // sum6n
								"eor    v30.16b, v30.16b, v30.16b    \n" // sum7
								"eor    v31.16b, v31.16b, v31.16b    \n" // sum7n

								"lsr         w4, %w20, #2            \n" // r4 = nn = L >> 2
								"cmp         w4, #0                  \n"
								"beq         1f                      \n"

								"0:                                  \n" // for (; k+3<L; k=k+4)

								"prfm   pldl1keep, [%9, #128]                       \n"
								"ld1    {v0.8b, v1.8b, v2.8b, v3.8b}, [%9], #32     \n"

								"prfm   pldl1keep, [%8, #128]                       \n"
								"ld1    {v8.8b, v9.8b, v10.8b, v11.8b}, [%8], #32   \n"

								"sshll    v0.8h, v0.8b, #0           \n" // k00 - k70
								"sshll    v1.8h, v1.8b, #0           \n" // k01 - k71
								"sshll    v2.8h, v2.8b, #0           \n" // k02 - k72
								"sshll    v3.8h, v3.8b, #0           \n" // k03 - k73

								"sshll    v8.8h, v8.8b, #0           \n" // a00 - a70
								"sshll    v9.8h, v9.8b, #0           \n" // a01 - a71
								"sshll    v10.8h, v10.8b, #0         \n" // a02 - a72
								"sshll    v11.8h, v11.8b, #0         \n" // a03 - a73
								// k0
								"smlal    v16.4s, v8.4h, v0.h[0]     \n" // sum0 += (a00-a70) * k00
								"smlal2   v17.4s, v8.8h, v0.h[0]     \n" //
								"smlal    v18.4s, v8.4h, v0.h[1]     \n" // sum1 += (a00-a70) * k10
								"smlal2   v19.4s, v8.8h, v0.h[1]     \n" //
								"smlal    v20.4s, v8.4h, v0.h[2]     \n" // sum2 += (a00-a70) * k20
								"smlal2   v21.4s, v8.8h, v0.h[2]     \n" //
								"smlal    v22.4s, v8.4h, v0.h[3]     \n" // sum3 += (a00-a70) * k30
								"smlal2   v23.4s, v8.8h, v0.h[3]     \n" //
								"smlal    v24.4s, v8.4h, v0.h[4]     \n" // sum4 += (a00-a70) * k40
								"smlal2   v25.4s, v8.8h, v0.h[4]     \n" //
								"smlal    v26.4s, v8.4h, v0.h[5]     \n" // sum5 += (a00-a70) * k50
								"smlal2   v27.4s, v8.8h, v0.h[5]     \n" //
								"smlal    v28.4s, v8.4h, v0.h[6]     \n" // sum6 += (a00-a70) * k60
								"smlal2   v29.4s, v8.8h, v0.h[6]     \n" //
								"smlal    v30.4s, v8.4h, v0.h[7]     \n" // sum7 += (a00-a70) * k70
								"smlal2   v31.4s, v8.8h, v0.h[7]     \n" //
								// k1
								"smlal    v16.4s, v9.4h, v1.h[0]     \n" // sum0 += (a01-a71) * k01
								"smlal2   v17.4s, v9.8h, v1.h[0]     \n" //
								"smlal    v18.4s, v9.4h, v1.h[1]     \n" // sum1 += (a01-a71) * k11
								"smlal2   v19.4s, v9.8h, v1.h[1]     \n" //
								"smlal    v20.4s, v9.4h, v1.h[2]     \n" // sum2 += (a01-a71) * k21
								"smlal2   v21.4s, v9.8h, v1.h[2]     \n" //
								"smlal    v22.4s, v9.4h, v1.h[3]     \n" // sum3 += (a01-a71) * k31
								"smlal2   v23.4s, v9.8h, v1.h[3]     \n" //
								"smlal    v24.4s, v9.4h, v1.h[4]     \n" // sum4 += (a01-a71) * k41
								"smlal2   v25.4s, v9.8h, v1.h[4]     \n" //
								"smlal    v26.4s, v9.4h, v1.h[5]     \n" // sum5 += (a01-a71) * k51
								"smlal2   v27.4s, v9.8h, v1.h[5]     \n" //
								"smlal    v28.4s, v9.4h, v1.h[6]     \n" // sum6 += (a01-a71) * k61
								"smlal2   v29.4s, v9.8h, v1.h[6]     \n" //
								"smlal    v30.4s, v9.4h, v1.h[7]     \n" // sum7 += (a01-a71) * k71
								"smlal2   v31.4s, v9.8h, v1.h[7]     \n" //
								// k2
								"smlal    v16.4s, v10.4h, v2.h[0]    \n" // sum0 += (a02-a72) * k02
								"smlal2   v17.4s, v10.8h, v2.h[0]    \n" //
								"smlal    v18.4s, v10.4h, v2.h[1]    \n" // sum1 += (a02-a72) * k12
								"smlal2   v19.4s, v10.8h, v2.h[1]    \n" //
								"smlal    v20.4s, v10.4h, v2.h[2]    \n" // sum2 += (a02-a72) * k22
								"smlal2   v21.4s, v10.8h, v2.h[2]    \n" //
								"smlal    v22.4s, v10.4h, v2.h[3]    \n" // sum3 += (a02-a72) * k32
								"smlal2   v23.4s, v10.8h, v2.h[3]    \n" //
								"smlal    v24.4s, v10.4h, v2.h[4]    \n" // sum4 += (a02-a72) * k42
								"smlal2   v25.4s, v10.8h, v2.h[4]    \n" //
								"smlal    v26.4s, v10.4h, v2.h[5]    \n" // sum5 += (a02-a72) * k52
								"smlal2   v27.4s, v10.8h, v2.h[5]    \n" //
								"smlal    v28.4s, v10.4h, v2.h[6]    \n" // sum6 += (a02-a72) * k62
								"smlal2   v29.4s, v10.8h, v2.h[6]    \n" //
								"smlal    v30.4s, v10.4h, v2.h[7]    \n" // sum7 += (a02-a72) * k72
								"smlal2   v31.4s, v10.8h, v2.h[7]    \n" //
								// k3
								"smlal    v16.4s, v11.4h, v3.h[0]    \n" // sum0 += (a03-a73) * k03
								"smlal2   v17.4s, v11.8h, v3.h[0]    \n" //
								"smlal    v18.4s, v11.4h, v3.h[1]    \n" // sum1 += (a03-a73) * k13
								"smlal2   v19.4s, v11.8h, v3.h[1]    \n" //
								"smlal    v20.4s, v11.4h, v3.h[2]    \n" // sum2 += (a03-a73) * k23
								"smlal2   v21.4s, v11.8h, v3.h[2]    \n" //
								"smlal    v22.4s, v11.4h, v3.h[3]    \n" // sum3 += (a03-a73) * k33
								"smlal2   v23.4s, v11.8h, v3.h[3]    \n" //
								"smlal    v24.4s, v11.4h, v3.h[4]    \n" // sum4 += (a03-a73) * k43
								"smlal2   v25.4s, v11.8h, v3.h[4]    \n" //
								"smlal    v26.4s, v11.4h, v3.h[5]    \n" // sum5 += (a03-a73) * k53
								"smlal2   v27.4s, v11.8h, v3.h[5]    \n" //
								"smlal    v28.4s, v11.4h, v3.h[6]    \n" // sum6 += (a03-a73) * k63
								"smlal2   v29.4s, v11.8h, v3.h[6]    \n" //
								"smlal    v30.4s, v11.4h, v3.h[7]    \n" // sum7 += (a03-a73) * k73
								"smlal2   v31.4s, v11.8h, v3.h[7]    \n" //

								"subs   w4, w4, #1                   \n"
								"bne    0b                           \n"

								"1:                                  \n"

								// remain loop
								"and    w4, %w20, #3                 \n" // w4 = remain = inch & 3;
								"cmp    w4, #0                       \n"
								"beq    3f                           \n"

								"2:                                  \n"

								"prfm   pldl1keep, [%9, #128]        \n"
								"ld1    {v0.8b}, [%9], #8            \n"

								"prfm   pldl1keep, [%8, #128]        \n"
								"ld1    {v8.8b}, [%8], #8            \n"

								"sshll    v0.8h, v0.8b, #0           \n" // k00 - k70
								"sshll    v8.8h, v8.8b, #0           \n" // a00 - a70

								// k0
								"smlal    v16.4s, v8.4h, v0.h[0]     \n" // sum0 += (a00-a70) * k00
								"smlal2   v17.4s, v8.8h, v0.h[0]     \n" //
								"smlal    v18.4s, v8.4h, v0.h[1]     \n" // sum1 += (a00-a70) * k10
								"smlal2   v19.4s, v8.8h, v0.h[1]     \n" //
								"smlal    v20.4s, v8.4h, v0.h[2]     \n" // sum2 += (a00-a70) * k20
								"smlal2   v21.4s, v8.8h, v0.h[2]     \n" //
								"smlal    v22.4s, v8.4h, v0.h[3]     \n" // sum3 += (a00-a70) * k30
								"smlal2   v23.4s, v8.8h, v0.h[3]     \n" //
								"smlal    v24.4s, v8.4h, v0.h[4]     \n" // sum4 += (a00-a70) * k40
								"smlal2   v25.4s, v8.8h, v0.h[4]     \n" //
								"smlal    v26.4s, v8.4h, v0.h[5]     \n" // sum5 += (a00-a70) * k50
								"smlal2   v27.4s, v8.8h, v0.h[5]     \n" //
								"smlal    v28.4s, v8.4h, v0.h[6]     \n" // sum6 += (a00-a70) * k60
								"smlal2   v29.4s, v8.8h, v0.h[6]     \n" //
								"smlal    v30.4s, v8.4h, v0.h[7]     \n" // sum7 += (a00-a70) * k70
								"smlal2   v31.4s, v8.8h, v0.h[7]     \n" //

								"subs   w4, w4, #1                   \n"

								"bne    2b                           \n"

								"3:                                  \n"

								"st1    {v16.4s, v17.4s}, [%0]       \n"
								"st1    {v18.4s, v19.4s}, [%1]       \n"
								"st1    {v20.4s, v21.4s}, [%2]       \n"
								"st1    {v22.4s, v23.4s}, [%3]       \n"
								"st1    {v24.4s, v25.4s}, [%4]       \n"
								"st1    {v26.4s, v27.4s}, [%5]       \n"
								"st1    {v28.4s, v29.4s}, [%6]       \n"
								"st1    {v30.4s, v31.4s}, [%7]       \n"

								: "=r"(output0), // %0
								"=r"(output1), // %1
								"=r"(output2), // %2
								"=r"(output3), // %3
								"=r"(output4), // %4
								"=r"(output5), // %5
								"=r"(output6), // %6
								"=r"(output7), // %7
								"=r"(vb),      // %8
								"=r"(va)       // %9
								: "0"(output0),
								"1"(output1),
								"2"(output2),
								"3"(output3),
								"4"(output4),
								"5"(output5),
								"6"(output6),
								"7"(output7),
								"8"(vb),
								"9"(va),
								"r"(L) // %20
								: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else
							int sum0[8] = { 0 };
							int sum1[8] = { 0 };
							int sum2[8] = { 0 };
							int sum3[8] = { 0 };
							int sum4[8] = { 0 };
							int sum5[8] = { 0 };
							int sum6[8] = { 0 };
							int sum7[8] = { 0 };

							int k = 0;
							for (; k + 7 < L; k = k + 8)
							{
								for (int n = 0; n < 8; n++)
								{
									sum0[n] += (int)va[0] * vb[n];
									sum1[n] += (int)va[1] * vb[n];
									sum2[n] += (int)va[2] * vb[n];
									sum3[n] += (int)va[3] * vb[n];
									sum4[n] += (int)va[4] * vb[n];
									sum5[n] += (int)va[5] * vb[n];
									sum6[n] += (int)va[6] * vb[n];
									sum7[n] += (int)va[7] * vb[n];
									va += 8;

									sum0[n] += (int)va[0] * vb[n + 8];
									sum1[n] += (int)va[1] * vb[n + 8];
									sum2[n] += (int)va[2] * vb[n + 8];
									sum3[n] += (int)va[3] * vb[n + 8];
									sum4[n] += (int)va[4] * vb[n + 8];
									sum5[n] += (int)va[5] * vb[n + 8];
									sum6[n] += (int)va[6] * vb[n + 8];
									sum7[n] += (int)va[7] * vb[n + 8];
									va += 8;

									sum0[n] += (int)va[0] * vb[n + 16];
									sum1[n] += (int)va[1] * vb[n + 16];
									sum2[n] += (int)va[2] * vb[n + 16];
									sum3[n] += (int)va[3] * vb[n + 16];
									sum4[n] += (int)va[4] * vb[n + 16];
									sum5[n] += (int)va[5] * vb[n + 16];
									sum6[n] += (int)va[6] * vb[n + 16];
									sum7[n] += (int)va[7] * vb[n + 16];
									va += 8;

									sum0[n] += (int)va[0] * vb[n + 24];
									sum1[n] += (int)va[1] * vb[n + 24];
									sum2[n] += (int)va[2] * vb[n + 24];
									sum3[n] += (int)va[3] * vb[n + 24];
									sum4[n] += (int)va[4] * vb[n + 24];
									sum5[n] += (int)va[5] * vb[n + 24];
									sum6[n] += (int)va[6] * vb[n + 24];
									sum7[n] += (int)va[7] * vb[n + 24];
									va += 8;

									sum0[n] += (int)va[0] * vb[n + 32];
									sum1[n] += (int)va[1] * vb[n + 32];
									sum2[n] += (int)va[2] * vb[n + 32];
									sum3[n] += (int)va[3] * vb[n + 32];
									sum4[n] += (int)va[4] * vb[n + 32];
									sum5[n] += (int)va[5] * vb[n + 32];
									sum6[n] += (int)va[6] * vb[n + 32];
									sum7[n] += (int)va[7] * vb[n + 32];
									va += 8;

									sum0[n] += (int)va[0] * vb[n + 40];
									sum1[n] += (int)va[1] * vb[n + 40];
									sum2[n] += (int)va[2] * vb[n + 40];
									sum3[n] += (int)va[3] * vb[n + 40];
									sum4[n] += (int)va[4] * vb[n + 40];
									sum5[n] += (int)va[5] * vb[n + 40];
									sum6[n] += (int)va[6] * vb[n + 40];
									sum7[n] += (int)va[7] * vb[n + 40];
									va += 8;

									sum0[n] += (int)va[0] * vb[n + 48];
									sum1[n] += (int)va[1] * vb[n + 48];
									sum2[n] += (int)va[2] * vb[n + 48];
									sum3[n] += (int)va[3] * vb[n + 48];
									sum4[n] += (int)va[4] * vb[n + 48];
									sum5[n] += (int)va[5] * vb[n + 48];
									sum6[n] += (int)va[6] * vb[n + 48];
									sum7[n] += (int)va[7] * vb[n + 48];
									va += 8;

									sum0[n] += (int)va[0] * vb[n + 56];
									sum1[n] += (int)va[1] * vb[n + 56];
									sum2[n] += (int)va[2] * vb[n + 56];
									sum3[n] += (int)va[3] * vb[n + 56];
									sum4[n] += (int)va[4] * vb[n + 56];
									sum5[n] += (int)va[5] * vb[n + 56];
									sum6[n] += (int)va[6] * vb[n + 56];
									sum7[n] += (int)va[7] * vb[n + 56];
									va -= 56;
								}

								va += 64;
								vb += 64;
							}

							for (; k < L; k++)
							{
								for (int n = 0; n < 8; n++)
								{
									sum0[n] += (int)va[0] * vb[n];
									sum1[n] += (int)va[1] * vb[n];
									sum2[n] += (int)va[2] * vb[n];
									sum3[n] += (int)va[3] * vb[n];
									sum4[n] += (int)va[4] * vb[n];
									sum5[n] += (int)va[5] * vb[n];
									sum6[n] += (int)va[6] * vb[n];
									sum7[n] += (int)va[7] * vb[n];
								}

								va += 8;
								vb += 8;
							}

							for (int n = 0; n < 8; n++)
							{
								output0[n] = sum0[n];
								output1[n] = sum1[n];
								output2[n] = sum2[n];
								output3[n] = sum3[n];
								output4[n] = sum4[n];
								output5[n] = sum5[n];
								output6[n] = sum6[n];
								output7[n] = sum7[n];
							}
#endif // __aarch64__
							output0 += 8;
							output1 += 8;
							output2 += 8;
							output3 += 8;
							output4 += 8;
							output5 += 8;
							output6 += 8;
							output7 += 8;
						}

						for (; j < N; j++)
						{
							signed char* vb = bottom_tm_data + (j / 8 + j % 8) * bottom_tm_cstep;
							const signed char* va = kernel_tm_int8_data + (i / 8) * kernel_tm_int8_cstep;

#if __aarch64__
							asm volatile(
								"eor    v14.16b, v14.16b, v14.16b    \n" // sum0_3
								"eor    v15.16b, v15.16b, v15.16b    \n" // sum4_7
								"eor    v16.16b, v16.16b, v16.16b    \n" // sum0
								"eor    v17.16b, v17.16b, v17.16b    \n" // sum1
								"eor    v18.16b, v18.16b, v18.16b    \n" // sum2
								"eor    v19.16b, v19.16b, v19.16b    \n" // sum3
								"eor    v20.16b, v20.16b, v20.16b    \n" // sum4
								"eor    v21.16b, v21.16b, v21.16b    \n" // sum5
								"eor    v22.16b, v22.16b, v22.16b    \n" // sum6
								"eor    v23.16b, v23.16b, v23.16b    \n" // sum7

								"lsr         w4, %w20, #2            \n" // r4 = nn = L >> 2
								"cmp         w4, #0                  \n"
								"beq         1f                      \n"

								"0:                                  \n" // for (; k+3<L; k=k+4)

								"prfm   pldl1keep, [%9, #128]                       \n"
								"ld1    {v0.8b, v1.8b, v2.8b, v3.8b}, [%9], #32     \n" // k

								//"prfm   pldl1keep, [%8, #128]      \n"
								"ld1    {v4.8b}, [%8]                \n" // d
								"add    %8, %8, #4                   \n"

								"sshll    v0.8h, v0.8b, #0           \n" // k00 - k70
								"sshll    v1.8h, v1.8b, #0           \n" // k01 - k71
								"sshll    v2.8h, v2.8b, #0           \n" // k02 - k72
								"sshll    v3.8h, v3.8b, #0           \n" // k03 - k73

								"sshll    v4.8h, v4.8b, #0           \n" // a00 - a30

								// k0
								"smlal    v16.4s, v0.4h, v4.h[0]     \n" // sum0 += (k00-k70) * a00
								"smlal2   v17.4s, v0.8h, v4.h[0]     \n" //
								"smlal    v18.4s, v1.4h, v4.h[1]     \n" // sum1 += (k01-k71) * a10
								"smlal2   v19.4s, v1.8h, v4.h[1]     \n" //
								"smlal    v20.4s, v2.4h, v4.h[2]     \n" // sum2 += (k02-k72) * a20
								"smlal2   v21.4s, v2.8h, v4.h[2]     \n" //
								"smlal    v22.4s, v3.4h, v4.h[3]     \n" // sum3 += (k03-k73) * a30
								"smlal2   v23.4s, v3.8h, v4.h[3]     \n" //

								"subs   w4, w4, #1                   \n"
								"bne    0b                           \n"

								"add      v16.4s, v16.4s, v18.4s     \n"
								"add      v17.4s, v17.4s, v19.4s     \n"
								"add      v20.4s, v20.4s, v22.4s     \n"
								"add      v21.4s, v21.4s, v23.4s     \n"
								"add      v14.4s, v16.4s, v20.4s     \n"
								"add      v15.4s, v17.4s, v21.4s     \n"

								"1:                                  \n"

								// remain loop
								"and    w4, %w20, #3                 \n" // w4 = remain = inch & 3;
								"cmp    w4, #0                       \n"
								"beq    3f                           \n"

								"2:                                  \n"

								//"prfm   pldl1keep, [%9, #128]      \n"
								"ld1    {v0.8b}, [%9], #8             \n"
								//"prfm   pldl1keep, [%8, #128]      \n"
								"ld1    {v4.8b}, [%8]                \n"
								"add    %8, %8, #1                   \n"

								"sshll    v0.8h, v0.8b, #0           \n" // k00 - k70
								"sshll    v4.8h, v4.8b, #0           \n" // a00

								// k0
								"smlal    v14.4s, v0.4h, v4.h[0]     \n" // sum0 += (k00-k70) * a00
								"smlal2   v15.4s, v0.8h, v4.h[0]     \n" //

								"subs   w4, w4, #1                   \n"

								"bne    2b                           \n"

								"3:                                  \n"

								"st1    {v14.s}[0], [%0]             \n"
								"st1    {v14.s}[1], [%1]             \n"
								"st1    {v14.s}[2], [%2]             \n"
								"st1    {v14.s}[3], [%3]             \n"
								"st1    {v15.s}[0], [%4]             \n"
								"st1    {v15.s}[1], [%5]             \n"
								"st1    {v15.s}[2], [%6]             \n"
								"st1    {v15.s}[3], [%7]             \n"

								: "=r"(output0), // %0
								"=r"(output1), // %1
								"=r"(output2), // %2
								"=r"(output3), // %3
								"=r"(output4), // %4
								"=r"(output5), // %5
								"=r"(output6), // %6
								"=r"(output7), // %7
								"=r"(vb),      // %8
								"=r"(va)       // %9
								: "0"(output0),
								"1"(output1),
								"2"(output2),
								"3"(output3),
								"4"(output4),
								"5"(output5),
								"6"(output6),
								"7"(output7),
								"8"(vb),
								"9"(va),
								"r"(L) // %20
								: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else
							int sum0 = 0;
							int sum1 = 0;
							int sum2 = 0;
							int sum3 = 0;
							int sum4 = 0;
							int sum5 = 0;
							int sum6 = 0;
							int sum7 = 0;

							for (int k = 0; k < L; k++)
							{
								sum0 += (int)va[0] * vb[0];
								sum1 += (int)va[1] * vb[0];
								sum2 += (int)va[2] * vb[0];
								sum3 += (int)va[3] * vb[0];
								sum4 += (int)va[4] * vb[0];
								sum5 += (int)va[5] * vb[0];
								sum6 += (int)va[6] * vb[0];
								sum7 += (int)va[7] * vb[0];

								va += 8;
								vb += 1;
							}

							output0[0] = sum0;
							output1[0] = sum1;
							output2[0] = sum2;
							output3[0] = sum3;
							output4[0] = sum4;
							output5[0] = sum5;
							output6[0] = sum6;
							output7[0] = sum7;
#endif // __aarch64__
							output0++;
							output1++;
							output2++;
							output3++;
							output4++;
							output5++;
							output6++;
							output7++;
				}
				}
#endif // __ARM_NEON && __aarch64__

					nn_outch = (outch - remain_outch_start) >> 2;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int pp = 0; pp < nn_outch; pp++)
					{
						int i = remain_outch_start + pp * 4;

						int* output0 = top_data + (i) * top_cstep;
						int* output1 = top_data + (i + 1) * top_cstep;
						int* output2 = top_data + (i + 2) * top_cstep;
						int* output3 = top_data + (i + 3) * top_cstep;

						int j = 0;
						for (; j + 7 < N; j = j + 8)
						{
							signed char* vb = bottom_tm_data + (j / 8) * bottom_tm_cstep;
#if __ARM_NEON && __aarch64__
							const signed char* va = kernel_tm_int8_data + (i / 8 + (i % 8) / 4) * kernel_tm_int8_cstep;
#else
							const signed char* va = kernel_tm_int8_data + (i / 4) * kernel_tm_int8_cstep;
#endif // __ARM_NEON && __aarch64__

#if __ARM_NEON
#if __aarch64__
							asm volatile(
								"eor    v16.16b, v16.16b, v16.16b    \n" // sum0
								"eor    v17.16b, v17.16b, v17.16b    \n" // sum0n
								"eor    v18.16b, v18.16b, v18.16b    \n" // sum1
								"eor    v19.16b, v19.16b, v19.16b    \n" // sum1n
								"eor    v20.16b, v20.16b, v20.16b    \n" // sum2
								"eor    v21.16b, v21.16b, v21.16b    \n" // sum2n
								"eor    v22.16b, v22.16b, v22.16b    \n" // sum3
								"eor    v23.16b, v23.16b, v23.16b    \n" // sum3n

								"lsr         w4, %w12, #2            \n" // r4 = nn = L >> 2
								"cmp         w4, #0                  \n"
								"beq         1f                      \n"

								"0:                                  \n" // for (; k+3<L; k=k+4)

								"prfm   pldl1keep, [%5, #128]        \n"
								"ld1    {v0.8b, v1.8b}, [%5], #16    \n"

								"prfm   pldl1keep, [%4, #128]                       \n"
								"ld1    {v8.8b, v9.8b, v10.8b, v11.8b}, [%4], #32   \n"

								"sshll    v0.8h, v0.8b, #0           \n" // k00 - k30,k01 - k31
								"sshll    v1.8h, v1.8b, #0           \n" // k02 - k32,k03 - k33

								"sshll    v8.8h, v8.8b, #0           \n" // a00 - a70
								"sshll    v9.8h, v9.8b, #0           \n" // a01 - a71
								"sshll    v10.8h, v10.8b, #0         \n" // a02 - a72
								"sshll    v11.8h, v11.8b, #0         \n" // a03 - a73

								// k0
								"smlal    v16.4s, v8.4h, v0.h[0]     \n" // sum0 += (a00-a70) * k00
								"smlal2   v17.4s, v8.8h, v0.h[0]     \n" //
								"smlal    v18.4s, v8.4h, v0.h[1]     \n" // sum1 += (a00-a70) * k10
								"smlal2   v19.4s, v8.8h, v0.h[1]     \n" //
								"smlal    v20.4s, v8.4h, v0.h[2]     \n" // sum2 += (a00-a70) * k20
								"smlal2   v21.4s, v8.8h, v0.h[2]     \n" //
								"smlal    v22.4s, v8.4h, v0.h[3]     \n" // sum3 += (a00-a70) * k30
								"smlal2   v23.4s, v8.8h, v0.h[3]     \n" //
								// k1
								"smlal    v16.4s, v9.4h, v0.h[4]     \n" // sum0 += (a01-a71) * k01
								"smlal2   v17.4s, v9.8h, v0.h[4]     \n" //
								"smlal    v18.4s, v9.4h, v0.h[5]     \n" // sum1 += (a01-a71) * k11
								"smlal2   v19.4s, v9.8h, v0.h[5]     \n" //
								"smlal    v20.4s, v9.4h, v0.h[6]     \n" // sum2 += (a01-a71) * k21
								"smlal2   v21.4s, v9.8h, v0.h[6]     \n" //
								"smlal    v22.4s, v9.4h, v0.h[7]     \n" // sum3 += (a01-a71) * k31
								"smlal2   v23.4s, v9.8h, v0.h[7]     \n" //
								// k2
								"smlal    v16.4s, v10.4h, v1.h[0]    \n" // sum0 += (a02-a72) * k02
								"smlal2   v17.4s, v10.8h, v1.h[0]    \n" //
								"smlal    v18.4s, v10.4h, v1.h[1]    \n" // sum1 += (a02-a72) * k12
								"smlal2   v19.4s, v10.8h, v1.h[1]    \n" //
								"smlal    v20.4s, v10.4h, v1.h[2]    \n" // sum2 += (a02-a72) * k22
								"smlal2   v21.4s, v10.8h, v1.h[2]    \n" //
								"smlal    v22.4s, v10.4h, v1.h[3]    \n" // sum3 += (a02-a72) * k32
								"smlal2   v23.4s, v10.8h, v1.h[3]    \n" //
								// k3
								"smlal    v16.4s, v11.4h, v1.h[4]    \n" // sum0 += (a03-a73) * k03
								"smlal2   v17.4s, v11.8h, v1.h[4]    \n" //
								"smlal    v18.4s, v11.4h, v1.h[5]    \n" // sum1 += (a03-a73) * k13
								"smlal2   v19.4s, v11.8h, v1.h[5]    \n" //
								"smlal    v20.4s, v11.4h, v1.h[6]    \n" // sum2 += (a03-a73) * k23
								"smlal2   v21.4s, v11.8h, v1.h[6]    \n" //
								"smlal    v22.4s, v11.4h, v1.h[7]    \n" // sum3 += (a03-a73) * k33
								"smlal2   v23.4s, v11.8h, v1.h[7]    \n" //

								"subs   w4, w4, #1                   \n"
								"bne    0b                           \n"

								"1:                                  \n"

								// remain loop
								"and    w4, %w12, #3                 \n" // w4 = remain = inch & 3;
								"cmp    w4, #0                       \n"
								"beq    3f                           \n"

								"2:                                  \n"

								//"prfm   pldl1keep, [%5, #128]      \n"
								"ld1    {v0.8b}, [%5]                \n"
								//"prfm   pldl1keep, [%4, #128]      \n"
								"ld1    {v8.8b}, [%4], #8            \n"
								"add    %5, %5, #4                   \n"

								"sshll    v0.8h, v0.8b, #0           \n" // k00 - k30
								"sshll    v8.8h, v8.8b, #0           \n" // a00 - a70

								// k0
								"smlal    v16.4s, v8.4h, v0.h[0]     \n" // sum0 += (a00-a70) * k00
								"smlal2   v17.4s, v8.8h, v0.h[0]     \n" //
								"smlal    v18.4s, v8.4h, v0.h[1]     \n" // sum1 += (a00-a70) * k10
								"smlal2   v19.4s, v8.8h, v0.h[1]     \n" //
								"smlal    v20.4s, v8.4h, v0.h[2]     \n" // sum2 += (a00-a70) * k20
								"smlal2   v21.4s, v8.8h, v0.h[2]     \n" //
								"smlal    v22.4s, v8.4h, v0.h[3]     \n" // sum3 += (a00-a70) * k30
								"smlal2   v23.4s, v8.8h, v0.h[3]     \n" //

								"subs   w4, w4, #1                   \n"

								"bne    2b                           \n"

								"3:                                  \n"

								"st1    {v16.4s, v17.4s}, [%0]       \n"
								"st1    {v18.4s, v19.4s}, [%1]       \n"
								"st1    {v20.4s, v21.4s}, [%2]       \n"
								"st1    {v22.4s, v23.4s}, [%3]       \n"

								: "=r"(output0), // %0
								"=r"(output1), // %1
								"=r"(output2), // %2
								"=r"(output3), // %3
								"=r"(vb),      // %4
								"=r"(va)       // %5
								: "0"(output0),
								"1"(output1),
								"2"(output2),
								"3"(output3),
								"4"(vb),
								"5"(va),
								"r"(L) // %12
								: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else
							asm volatile(
								// K loop
								"vmov.s32    q8, #0             \n"
								"vmov.s32    q9, #0             \n"
								"vmov.s32    q10, #0            \n"
								"vmov.s32    q11, #0            \n"
								"vmov.s32    q12, #0            \n"
								"vmov.s32    q13, #0            \n"
								"vmov.s32    q14, #0            \n"
								"vmov.s32    q15, #0            \n"

								"lsr         r4, %12, #3        \n" // r4 = nn = L >> 3
								"cmp         r4, #0             \n"
								"beq         1f                 \n"

								"0:                             \n" // for(; nn != 0; nn--)
								"pld         [%4, #128]         \n"
								"vld1.s8     {d8-d11}, [%4]!    \n" // tmpr a00-a07,a10-a17,a20-a27,a30-a37    a(inch)(data)
								"vmovl.s8    q7, d11            \n" // a30-a37
								"vmovl.s8    q6, d10            \n" // a20-a27
								"vmovl.s8    q5, d9             \n" // a10-a17
								"vmovl.s8    q4, d8             \n" // a00-a07

								"pld         [%5, #128]         \n"
								"vld1.s8     {d0-d3}, [%5]!     \n" // kptr k00-k30,k01-k31, k02-k32,k03-k33, k04-k34,k05-k35, k06-k36,k07-k37    k(outch)(inch)
								"vmovl.s8    q3, d3             \n" // k06-k36,k07-k37
								"vmovl.s8    q2, d2             \n" // k04-k34,k05-k35
								"vmovl.s8    q1, d1             \n" // k02-k32,k03-k33
								"vmovl.s8    q0, d0             \n" // k00-k30,k01-k31

								"vmlal.s16   q8, d8, d0[0]      \n" // sum0 = (a00-a07) * k00
								"vmlal.s16   q9, d9, d0[0]      \n"
								"vmlal.s16   q10, d8, d0[1]     \n" // sum1 = (a00-a07) * k10
								"vmlal.s16   q11, d9, d0[1]     \n"
								"vmlal.s16   q12, d8, d0[2]     \n" // sum2 = (a00-a07) * k20
								"vmlal.s16   q13, d9, d0[2]     \n"
								"vmlal.s16   q14, d8, d0[3]     \n" // sum3 = (a00-a07) * k30
								"vmlal.s16   q15, d9, d0[3]     \n"

								"vmlal.s16   q8, d10, d1[0]     \n" // sum0 += (a10-a17) * k01
								"vmlal.s16   q9, d11, d1[0]     \n"
								"vmlal.s16   q10, d10, d1[1]    \n" // sum1 += (a10-a17) * k11
								"vmlal.s16   q11, d11, d1[1]    \n"
								"vmlal.s16   q12, d10, d1[2]    \n" // sum2 += (a10-a17) * k21
								"vmlal.s16   q13, d11, d1[2]    \n"
								"vmlal.s16   q14, d10, d1[3]    \n" // sum3 += (a10-a17) * k31
								"vmlal.s16   q15, d11, d1[3]    \n"

								"pld         [%4, #128]         \n"
								"vld1.s8     {d8-d9}, [%4]!     \n" // tmpr a00-a07,a10-a17,a20-a27,a30-a37    a(inch)(data)
								"vmovl.s8    q5, d9             \n" // a10-a17
								"vmovl.s8    q4, d8             \n" // a00-a07

								"vmlal.s16   q8, d12, d2[0]     \n" // sum0 += (a20-a27) * k02
								"vmlal.s16   q9, d13, d2[0]     \n"
								"vmlal.s16   q10, d12, d2[1]    \n" // sum1 += (a20-a27) * k12
								"vmlal.s16   q11, d13, d2[1]    \n"
								"vmlal.s16   q12, d12, d2[2]    \n" // sum2 += (a20-a27) * k22
								"vmlal.s16   q13, d13, d2[2]    \n"
								"vmlal.s16   q14, d12, d2[3]    \n" // sum3 += (a20-a27) * k32
								"vmlal.s16   q15, d13, d2[3]    \n"

								"vmlal.s16   q8, d14, d3[0]     \n" // sum0 += (a30-a37) * k03
								"vmlal.s16   q9, d15, d3[0]     \n"
								"vmlal.s16   q10, d14, d3[1]    \n" // sum1 += (a30-a37) * k13
								"vmlal.s16   q11, d15, d3[1]    \n"
								"vmlal.s16   q12, d14, d3[2]    \n" // sum2 += (a30-a37) * k23
								"vmlal.s16   q13, d15, d3[2]    \n"
								"vmlal.s16   q14, d14, d3[3]    \n" // sum3 += (a30-a37) * k33
								"vmlal.s16   q15, d15, d3[3]    \n"

								"pld         [%4, #128]         \n"
								"vld1.s8     {d0-d1}, [%4]!     \n" // tmpr a00-a07,a10-a17,a20-a27,a30-a37    a(inch)(data)
								"vmovl.s8    q1, d1             \n" // a10-a17
								"vmovl.s8    q0, d0             \n" // a00-a07

								"vmlal.s16   q8, d8, d4[0]      \n" // sum0 += (a40-a47) * k04
								"vmlal.s16   q9, d9, d4[0]      \n"
								"vmlal.s16   q10, d8, d4[1]     \n" // sum1 += (a40-a47) * k14
								"vmlal.s16   q11, d9, d4[1]     \n"
								"vmlal.s16   q12, d8, d4[2]     \n" // sum2 += (a40-a47) * k24
								"vmlal.s16   q13, d9, d4[2]     \n"
								"vmlal.s16   q14, d8, d4[3]     \n" // sum3 += (a40-a47) * k34
								"vmlal.s16   q15, d9, d4[3]     \n"

								"vmlal.s16   q8, d10, d5[0]     \n" // sum0 += (a50-a57) * k05
								"vmlal.s16   q9, d11, d5[0]     \n"
								"vmlal.s16   q10, d10, d5[1]    \n" // sum1 += (a50-a57) * k15
								"vmlal.s16   q11, d11, d5[1]    \n"
								"vmlal.s16   q12, d10, d5[2]    \n" // sum2 += (a50-a57) * k25
								"vmlal.s16   q13, d11, d5[2]    \n"
								"vmlal.s16   q14, d10, d5[3]    \n" // sum3 += (a50-a57) * k35
								"vmlal.s16   q15, d11, d5[3]    \n"

								"vmlal.s16   q8, d0, d6[0]      \n" // sum0 += (a60-a67) * k06
								"vmlal.s16   q9, d1, d6[0]      \n"
								"vmlal.s16   q10, d0, d6[1]     \n" // sum1 += (a60-a67) * k16
								"vmlal.s16   q11, d1, d6[1]     \n"
								"vmlal.s16   q12, d0, d6[2]     \n" // sum2 += (a60-a67) * k26
								"vmlal.s16   q13, d1, d6[2]     \n"
								"vmlal.s16   q14, d0, d6[3]     \n" // sum3 += (a60-a67) * k36
								"vmlal.s16   q15, d1, d6[3]     \n"

								"vmlal.s16   q8, d2, d7[0]      \n" // sum0 += (a70-a77) * k07
								"vmlal.s16   q9, d3, d7[0]      \n"
								"vmlal.s16   q10, d2, d7[1]     \n" // sum1 += (a70-a77) * k17
								"vmlal.s16   q11, d3, d7[1]     \n"
								"vmlal.s16   q12, d2, d7[2]     \n" // sum2 += (a70-a77) * k27
								"vmlal.s16   q13, d3, d7[2]     \n"
								"vmlal.s16   q14, d2, d7[3]     \n" // sum3 += (a70-a77) * k37
								"vmlal.s16   q15, d3, d7[3]     \n"

								"subs        r4, r4, #1         \n"
								"bne         0b                 \n" // end for

								"1:                             \n"
								// remain loop
								"and         r4, %12, #7        \n" // r4 = remain = inch & 7
								"cmp         r4, #0             \n"
								"beq         3f                 \n"

								"2:                             \n" // for(; remain != 0; remain--)
								"vld1.s8     {d2}, [%4]!        \n" // tmpr a00-a70    a(inch)(data)
								"vld1.s8     {d0}, [%5]         \n" // kptr k00-k30    k(outch)(inch)
								"vmovl.s8    q1, d2             \n"
								"vmovl.s8    q0, d0             \n"
								"add         %5, #4             \n"

								"vmlal.s16   q8, d2, d0[0]      \n" // sum0 += (a00-a70) * k00
								"vmlal.s16   q9, d3, d0[0]      \n"
								"vmlal.s16   q10, d2, d0[1]     \n" // sum1 += (a00-a70) * k10
								"vmlal.s16   q11, d3, d0[1]     \n"
								"vmlal.s16   q12, d2, d0[2]     \n" // sum2 += (a00-a70) * k20
								"vmlal.s16   q13, d3, d0[2]     \n"
								"vmlal.s16   q14, d2, d0[3]     \n" // sum3 += (a00-a70) * k30
								"vmlal.s16   q15, d3, d0[3]     \n"

								"subs        r4, r4, #1         \n"
								"bne         2b                 \n"

								"3:                             \n" // store the result to memory
								"vst1.s32    {d16-d19}, [%0]    \n"
								"vst1.s32    {d20-d23}, [%1]    \n"
								"vst1.s32    {d24-d27}, [%2]    \n"
								"vst1.s32    {d28-d31}, [%3]    \n"

								: "=r"(output0), // %0
								"=r"(output1), // %1
								"=r"(output2), // %2
								"=r"(output3), // %3
								"=r"(vb),      // %4
								"=r"(va)       // %5
								: "0"(output0),
								"1"(output1),
								"2"(output2),
								"3"(output3),
								"4"(vb),
								"5"(va),
								"r"(L) // %12
								: "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else
							int sum0[8] = { 0 };
							int sum1[8] = { 0 };
							int sum2[8] = { 0 };
							int sum3[8] = { 0 };

							int k = 0;
							for (; k + 7 < L; k = k + 8)
							{
								for (int n = 0; n < 8; n++)
								{
									sum0[n] += (int)va[0] * vb[n];
									sum1[n] += (int)va[1] * vb[n];
									sum2[n] += (int)va[2] * vb[n];
									sum3[n] += (int)va[3] * vb[n];
									va += 4;

									sum0[n] += (int)va[0] * vb[n + 8];
									sum1[n] += (int)va[1] * vb[n + 8];
									sum2[n] += (int)va[2] * vb[n + 8];
									sum3[n] += (int)va[3] * vb[n + 8];
									va += 4;

									sum0[n] += (int)va[0] * vb[n + 16];
									sum1[n] += (int)va[1] * vb[n + 16];
									sum2[n] += (int)va[2] * vb[n + 16];
									sum3[n] += (int)va[3] * vb[n + 16];
									va += 4;

									sum0[n] += (int)va[0] * vb[n + 24];
									sum1[n] += (int)va[1] * vb[n + 24];
									sum2[n] += (int)va[2] * vb[n + 24];
									sum3[n] += (int)va[3] * vb[n + 24];
									va += 4;

									sum0[n] += (int)va[0] * vb[n + 32];
									sum1[n] += (int)va[1] * vb[n + 32];
									sum2[n] += (int)va[2] * vb[n + 32];
									sum3[n] += (int)va[3] * vb[n + 32];
									va += 4;

									sum0[n] += (int)va[0] * vb[n + 40];
									sum1[n] += (int)va[1] * vb[n + 40];
									sum2[n] += (int)va[2] * vb[n + 40];
									sum3[n] += (int)va[3] * vb[n + 40];
									va += 4;

									sum0[n] += (int)va[0] * vb[n + 48];
									sum1[n] += (int)va[1] * vb[n + 48];
									sum2[n] += (int)va[2] * vb[n + 48];
									sum3[n] += (int)va[3] * vb[n + 48];
									va += 4;

									sum0[n] += (int)va[0] * vb[n + 56];
									sum1[n] += (int)va[1] * vb[n + 56];
									sum2[n] += (int)va[2] * vb[n + 56];
									sum3[n] += (int)va[3] * vb[n + 56];
									va -= 28;
								}

								va += 32;
								vb += 64;
							}

							for (; k < L; k++)
							{
								for (int n = 0; n < 8; n++)
								{
									sum0[n] += (int)va[0] * vb[n];
									sum1[n] += (int)va[1] * vb[n];
									sum2[n] += (int)va[2] * vb[n];
									sum3[n] += (int)va[3] * vb[n];
								}

								va += 4;
								vb += 8;
							}

							for (int n = 0; n < 8; n++)
							{
								output0[n] = sum0[n];
								output1[n] = sum1[n];
								output2[n] = sum2[n];
								output3[n] = sum3[n];
							}
#endif // __ARM_NEON
							output0 += 8;
							output1 += 8;
							output2 += 8;
							output3 += 8;
						}

						for (; j < N; j++)
						{
							signed char* vb = bottom_tm.channel(j / 8 + j % 8);
#if __ARM_NEON && __aarch64__
							const signed char* va = kernel_tm_int8_data + (i / 8 + (i % 8) / 4) * kernel_tm_int8_cstep;
#else
							const signed char* va = kernel_tm_int8_data + (i / 4) * kernel_tm_int8_cstep;
#endif // __ARM_NEON && __aarch64__

#if __ARM_NEON
#if __aarch64__
							asm volatile(
								"eor    v14.16b, v14.16b, v14.16b    \n" // sum0_3
								"eor    v16.16b, v16.16b, v16.16b    \n" // sum0
								"eor    v17.16b, v17.16b, v17.16b    \n" // sum1
								"eor    v18.16b, v18.16b, v18.16b    \n" // sum2
								"eor    v19.16b, v19.16b, v19.16b    \n" // sum3

								"lsr         w4, %w12, #2            \n" // r4 = nn = L >> 2
								"cmp         w4, #0                  \n"
								"beq         1f                      \n"

								"0:                                  \n" // for (; k+3<L; k=k+4)

								"prfm   pldl1keep, [%5, #128]        \n"
								"ld1    {v0.8b, v1.8b}, [%5], #16    \n" // k

								//"prfm   pldl1keep, [%4, #128]      \n"
								"ld1    {v4.8b}, [%4]                \n" // d
								"add    %4, %4, #4                   \n"

								"sshll    v0.8h, v0.8b, #0           \n" // k00 - k30,k01 - k31
								"sshll    v1.8h, v1.8b, #0           \n" // k02 - k32,k03 - k33
								"sshll    v4.8h, v4.8b, #0           \n" // a00 - a30

								"subs   w4, w4, #1                   \n"
								// k0
								"smlal    v16.4s, v0.4h, v4.h[0]     \n" // sum0 += (k00-k30) * a00
								"smlal2   v17.4s, v0.8h, v4.h[0]     \n" // sum1 += (k01-k31) * a10
								"smlal    v18.4s, v1.4h, v4.h[1]     \n" // sum2 += (k02-k32) * a20
								"smlal2   v19.4s, v1.8h, v4.h[1]     \n" // sum3 += (k03-k33) * a30

								"bne    0b                           \n"

								"add      v16.4s, v16.4s, v18.4s     \n"
								"add      v17.4s, v17.4s, v19.4s     \n"
								"add      v14.4s, v16.4s, v17.4s     \n"

								"1:                                  \n"

								// remain loop
								"and    w4, %w12, #3                 \n" // w4 = remain = inch & 3;
								"cmp    w4, #0                       \n"
								"beq    3f                           \n"

								"2:                                  \n"

								//"prfm   pldl1keep, [%5, #128]      \n"
								"ld1    {v0.8b}, [%5]                \n"
								//"prfm   pldl1keep, [4, #128]       \n"
								"ld1    {v4.8b}, [%4]                \n"
								"add    %4, %4, #1                   \n"
								"add    %5, %5, #4                   \n"

								"subs   w4, w4, #1                   \n"

								"sshll    v0.8h, v0.8b, #0           \n" // k00 - k30
								"sshll    v4.8h, v4.8b, #0           \n" // a00
								// k0
								"smlal    v14.4s, v0.4h, v4.h[0]     \n" // sum0 += (k00-k30) * a00

								"bne    2b                           \n"

								"3:                                  \n"

								"st1    {v14.s}[0], [%0]             \n"
								"st1    {v14.s}[1], [%1]             \n"
								"st1    {v14.s}[2], [%2]             \n"
								"st1    {v14.s}[3], [%3]             \n"

								: "=r"(output0), // %0
								"=r"(output1), // %1
								"=r"(output2), // %2
								"=r"(output3), // %3
								"=r"(vb),      // %4
								"=r"(va)       // %5
								: "0"(output0),
								"1"(output1),
								"2"(output2),
								"3"(output3),
								"4"(vb),
								"5"(va),
								"r"(L) // %12
								: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
#else
							asm volatile(
								// inch loop
								"veor        q6, q6, q6        \n"
								"veor        q7, q7, q7        \n"
								"veor        q8, q8, q8        \n"
								"veor        q9, q9, q9        \n"
								"veor        q10, q10, q10     \n"
								"veor        q11, q11, q11     \n"
								"veor        q12, q12, q12     \n"
								"veor        q13, q13, q13     \n"
								"vmov.s32    q14, #0           \n"

								"lsr         r4, %12, #3       \n" // r4 = nn = L >> 2
								"cmp         r4, #0            \n"
								"beq         1f                \n"

								"0:                            \n" // for(; nn != 0; nn--)
								"pld         [%4, #128]        \n"
								"vld1.s8     {d0}, [%4]!       \n" // tmpr a00,a10,a20,a30    a(inch)(data)
								"vmovl.s8    q0, d0            \n" // a00-a07

								"pld         [%5, #128]        \n"
								"vld1.s8     {d2-d5}, [%5]!    \n" // kptr k00-k30,k01-k31, k02-k32,k03-k33, k04-k34,k05-k35, k06-k36,k07-k37    k(outch)(inch)
								"vmovl.s8    q4, d5            \n" // k06-k36,k07-k37
								"vmovl.s8    q3, d4            \n" // k04-k34,k05-k35
								"vmovl.s8    q2, d3            \n" // k02-k32,k03-k33
								"vmovl.s8    q1, d2            \n" // k00-k30,k01-k31

								"vmlal.s16   q6, d2, d0[0]     \n" // (k00-k30) * a00
								"vmlal.s16   q7, d3, d0[1]     \n" // (k01-k31) * a01
								"vmlal.s16   q8, d4, d0[2]     \n" // (k02-k32) * a02
								"vmlal.s16   q9, d5, d0[3]     \n" // (k03-k33) * a03
								"vmlal.s16   q10, d6, d1[0]    \n" // (k04-k34) * a04
								"vmlal.s16   q11, d7, d1[1]    \n" // (k05-k35) * a05
								"vmlal.s16   q12, d8, d1[2]    \n" // (k06-k36) * a06
								"vmlal.s16   q13, d9, d1[3]    \n" // (k07-k37) * a07

								"subs        r4, r4, #1        \n"
								"bne         0b                \n" // end for

								"vadd.s32    q6, q6, q7        \n"
								"vadd.s32    q9, q9, q8        \n"
								"vadd.s32    q11, q11, q10     \n"
								"vadd.s32    q13, q13, q12     \n"

								"vadd.s32    q9, q9, q6        \n"
								"vadd.s32    q13, q13, q11     \n"
								"vadd.s32    q14, q13, q9      \n"

								"1:                            \n"
								// remain loop
								"and         r4, %12, #7       \n" // r4 = remain = inch & 3
								"cmp         r4, #0            \n"
								"beq         3f                \n"

								"2:                            \n" // for(; remain != 0; remain--)
								"vld1.s8     {d2}, [%4]        \n" // tmpr a00        a(inch)(data)
								"vld1.s8     {d0}, [%5]        \n" // kptr k00-k30    k(outch)(inch)
								"vmovl.s8    q1, d2            \n"
								"vmovl.s8    q0, d0            \n"
								"add         %4, #1            \n"
								"add         %5, #4            \n"

								"vmlal.s16   q14, d0, d2[0]    \n"

								"subs        r4, r4, #1        \n"
								"bne         2b                \n"

								"3:                            \n" // store the result to memory
								"vst1.s32    {d28[0]}, [%0]    \n"
								"vst1.s32    {d28[1]}, [%1]    \n"
								"vst1.s32    {d29[0]}, [%2]    \n"
								"vst1.s32    {d29[1]}, [%3]    \n"

								: "=r"(output0), // %0
								"=r"(output1), // %1
								"=r"(output2), // %2
								"=r"(output3), // %3
								"=r"(vb),      // %4
								"=r"(va)       // %5
								: "0"(output0),
								"1"(output1),
								"2"(output2),
								"3"(output3),
								"4"(vb),
								"5"(va),
								"r"(L) // %12
								: "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14");
#endif // __aarch64__
#else
							int sum0 = 0;
							int sum1 = 0;
							int sum2 = 0;
							int sum3 = 0;

							for (int k = 0; k < L; k++)
							{
								sum0 += (int)va[0] * vb[0];
								sum1 += (int)va[1] * vb[0];
								sum2 += (int)va[2] * vb[0];
								sum3 += (int)va[3] * vb[0];

								va += 4;
								vb += 1;
							}

							output0[0] = sum0;
							output1[0] = sum1;
							output2[0] = sum2;
							output3[0] = sum3;
#endif // __ARM_NEON
							output0++;
							output1++;
							output2++;
							output3++;
						}
					}

					remain_outch_start += nn_outch << 2;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
					for (int i = remain_outch_start; i < outch; i++)
					{
						int* output = top_data + (i) * top_cstep;

						int j = 0;
						for (; j + 7 < N; j = j + 8)
						{
							signed char* vb = bottom_tm_data + (j / 8) * bottom_tm_cstep;
#if __ARM_NEON && __aarch64__
							const signed char* va = kernel_tm_int8_data + (i / 8 + (i % 8) / 4 + i % 4) * kernel_tm_int8_cstep;
#else
							const signed char* va = kernel_tm_int8_data + (i / 4 + i % 4) * kernel_tm_int8_cstep;
#endif // __ARM_NEON && __aarch64__

#if __ARM_NEON
#if __aarch64__
							asm volatile(
								"eor    v16.16b, v16.16b, v16.16b    \n" // sum0
								"eor    v17.16b, v17.16b, v17.16b    \n" // sum0n

								"lsr         w4, %w6, #2             \n" // r4 = nn = L >> 2
								"cmp         w4, #0                  \n"
								"beq         1f                      \n"

								"0:                                  \n" // for (; k+3<L; k=k+4)

								"prfm   pldl1keep, [%2, #128]        \n"
								"ld1    {v0.8b}, [%2]                \n"

								"prfm   pldl1keep, [%1, #128]                       \n"
								"ld1    {v8.8b, v9.8b, v10.8b, v11.8b}, [%1], #32   \n"
								"add    %2, %2, #4                   \n"

								"sshll    v0.8h, v0.8b, #0           \n" // k00 - k03

								"sshll    v8.8h, v8.8b, #0           \n" // a00 - a70
								"sshll    v9.8h, v9.8b, #0           \n" // a01 - a71
								"sshll    v10.8h, v10.8b, #0         \n" // a02 - a72
								"sshll    v11.8h, v11.8b, #0         \n" // a03 - a73

								// k0
								"smlal    v16.4s, v8.4h, v0.h[0]     \n" // sum0 += (a00-a70) * k00
								"smlal2   v17.4s, v8.8h, v0.h[0]     \n" //
								// k1
								"smlal    v16.4s, v9.4h, v0.h[1]     \n" // sum0 += (a01-a71) * k01
								"smlal2   v17.4s, v9.8h, v0.h[1]     \n" //
								// k2
								"smlal    v16.4s, v10.4h, v0.h[2]    \n" // sum0 += (a02-a72) * k02
								"smlal2   v17.4s, v10.8h, v0.h[2]    \n" //
								// k3
								"smlal    v16.4s, v11.4h, v0.h[3]    \n" // sum0 += (a03-a73) * k03
								"smlal2   v17.4s, v11.8h, v0.h[3]    \n" //

								"subs   w4, w4, #1                   \n"
								"bne    0b                           \n"

								"1:                                  \n"

								// remain loop
								"and    w4, %w6, #3                 \n" // w4 = remain = inch & 3;
								"cmp    w4, #0                       \n"
								"beq    3f                           \n"

								"2:                                  \n"

								//"prfm   pldl1keep, [%2, #128]      \n"
								"ld1    {v0.8b}, [%2]                \n"
								//"prfm   pldl1keep, [%1, #128]      \n"
								"ld1    {v8.8b}, [%1], #8            \n"
								"add    %2, %2, #1                   \n"

								"sshll    v0.8h, v0.8b, #0           \n" // k00 - k30
								"sshll    v8.8h, v8.8b, #0           \n" // a00 - a70

								// k0
								"smlal    v16.4s, v8.4h, v0.h[0]     \n" // sum0 += (a00-a70) * k00
								"smlal2   v17.4s, v8.8h, v0.h[0]     \n" //

								"subs   w4, w4, #1                   \n"

								"bne    2b                           \n"

								"3:                                  \n"

								"st1    {v16.4s, v17.4s}, [%0]       \n"

								: "=r"(output), // %0
								"=r"(vb),     // %1
								"=r"(va)      // %2
								: "0"(output),
								"1"(vb),
								"2"(va),
								"r"(L) // %6
								: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17");
#else
							asm volatile(
								// inch loop
								"vmov.s32    q6, #0            \n"
								"vmov.s32    q7, #0            \n"

								"lsr         r4, %6, #3        \n" // r4 = nn = inch >> 3
								"cmp         r4, #0            \n"
								"beq         1f                \n"

								"0:                            \n" // for(; nn != 0; nn--)
								"pld         [%1, #128]        \n"
								"vld1.s8     {d4-d7}, [%1]!    \n" // tmpr a00-a07,a10-a17,a20-a27,a30-a37    a(inch)(data)
								"vmovl.s8    q5, d7            \n" // a30-a37
								"vmovl.s8    q4, d6            \n" // a20-a27
								"vmovl.s8    q3, d5            \n" // a10-a17
								"vmovl.s8    q2, d4            \n" // a00-a07

								"pld         [%2, #128]        \n"
								"vld1.s8     {d0}, [%2]!       \n" // kptr k00-k07    k(outch)(inch)
								"vmovl.s8    q1, d1            \n" // k04,k05,k06,k07
								"vmovl.s8    q0, d0            \n" // k00,k01,k02,k03

								"vmlal.s16   q6, d4, d0[0]     \n" // (a00-a07) * k00
								"vmlal.s16   q7, d5, d0[0]     \n"
								"vmlal.s16   q6, d6, d0[1]     \n" // (a10-a17) * k01
								"vmlal.s16   q7, d7, d0[1]     \n"
								"vmlal.s16   q6, d8, d0[2]     \n" // (a20-a27) * k02
								"vmlal.s16   q7, d9, d0[2]     \n"
								"vmlal.s16   q6, d10, d0[3]    \n" // (a30-a37) * k03
								"vmlal.s16   q7, d11, d0[3]    \n"

								"pld         [%1, #128]        \n"
								"vld1.s8     {d4-d7}, [%1]!    \n" // tmpr a40-a47,a50-a57,a60-a67,a70-a77    a(inch)(data)
								"vmovl.s8    q5, d7            \n" // a70-a77
								"vmovl.s8    q4, d6            \n" // a60-a67
								"vmovl.s8    q3, d5            \n" // a50-a57
								"vmovl.s8    q2, d4            \n" // a40-a47

								"vmlal.s16   q6, d4, d1[0]     \n" // (a00-a07) * k00
								"vmlal.s16   q7, d5, d1[0]     \n"
								"vmlal.s16   q6, d6, d1[1]     \n" // (a10-a17) * k01
								"vmlal.s16   q7, d7, d1[1]     \n"
								"vmlal.s16   q6, d8, d1[2]     \n" // (a20-a27) * k02
								"vmlal.s16   q7, d9, d1[2]     \n"
								"vmlal.s16   q6, d10, d1[3]    \n" // (a30-a37) * k03
								"vmlal.s16   q7, d11, d1[3]    \n"

								"subs        r4, r4, #1        \n"
								"bne         0b                \n" // end for

								"1:                            \n"
								// remain loop
								"and         r4, %6, #7        \n" // r4 = remain = inch & 7
								"cmp         r4, #0            \n"
								"beq         3f                \n"

								"2:                            \n" // for(; remain != 0; remain--)
								"vld1.s8     {d2}, [%1]!       \n" // tmpr a00-a07    a(inch)(data)
								"vld1.s8     {d0}, [%2]        \n" // kptr k00        k(outch)(inch)
								"vmovl.s8    q1, d2            \n"
								"vmovl.s8    q0, d0            \n"
								"add         %2, #1            \n"

								"vmlal.s16   q6, d2, d0[0]     \n" // (a00-a07) * k00
								"vmlal.s16   q7, d3, d0[0]     \n"

								"subs        r4, r4, #1        \n"
								"bne         2b                \n"

								"3:                            \n" // store the result to memory
								"vst1.s32    {d12-d15}, [%0]   \n"

								: "=r"(output), // %0
								"=r"(vb),     // %1
								"=r"(va)      // %2
								: "0"(output),
								"1"(vb),
								"2"(va),
								"r"(L) // %6
								: "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
#endif // __aarch64__
#else
							int sum[8] = { 0 };

							int k = 0;
							for (; k + 7 < L; k = k + 8)
							{
								for (int n = 0; n < 8; n++)
								{
									sum[n] += (int)va[0] * vb[n];
									sum[n] += (int)va[1] * vb[n + 8];
									sum[n] += (int)va[2] * vb[n + 16];
									sum[n] += (int)va[3] * vb[n + 24];
									sum[n] += (int)va[4] * vb[n + 32];
									sum[n] += (int)va[5] * vb[n + 40];
									sum[n] += (int)va[6] * vb[n + 48];
									sum[n] += (int)va[7] * vb[n + 56];
								}

								va += 8;
								vb += 64;
							}

							for (; k < L; k++)
							{
								for (int n = 0; n < 8; n++)
								{
									sum[n] += (int)va[0] * vb[n];
								}

								va += 1;
								vb += 8;
							}

							for (int n = 0; n < 8; n++)
							{
								output[n] = sum[n];
							}
#endif // __ARM_NEON
							output += 8;
						}

						for (; j < N; j++)
						{
							int sum = 0;

							signed char* vb = bottom_tm_data + (j / 8 + j % 8) * bottom_tm_cstep;
#if __ARM_NEON && __aarch64__
							const signed char* va = kernel_tm_int8_data + (i / 8 + (i % 8) / 4 + i % 4) * kernel_tm_int8_cstep;
#else
							const signed char* va = kernel_tm_int8_data + (i / 4 + i % 4) * kernel_tm_int8_cstep;
#endif // __ARM_NEON && __aarch64__

							for (int k = 0; k < L; k++)
							{
								sum += (int)va[0] * vb[0];

								va += 1;
								vb += 1;
							}
							output[0] = sum;

							output++;
						}
					}
				}
			}
		}
#endif

		INSTANCE_CLASS(operation_convolution_arm);
		REGISTE(operation_convolution_arm);
	}
}