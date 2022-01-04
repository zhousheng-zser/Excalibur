#include "../../../include/Excalibur/arm/operation_convolutiondepthwise_arm.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"
#include "../../../include/Excalibur/operation_make_border.hpp"
#include "../../../include/Excalibur/operation_cut_border.hpp"
#include <random>

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_convolutiondepthwise_arm<Dtype>::operation_convolutiondepthwise_arm(const operation_param& param) : operation_general_conv<Dtype>(param)
		{
			op = new operation_convolutiondepthwise<float>(param);
		}

		template<typename Dtype>
		operation_convolutiondepthwise_arm<Dtype>::~operation_convolutiondepthwise_arm()
		{
			delete op;
		}

		template<typename Dtype>
		int operation_convolutiondepthwise_arm<Dtype>::init_weights()
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
			}

			op->init_weights();
			return mem;
		}

		template<typename Dtype>
		int operation_convolutiondepthwise_arm<Dtype>::init_weights(FILE * fp)
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
			}
			else if (quantize_tag == 871224)
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
				mem += (this->output_channel_ + 1)  * sizeof(float);
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			fseek(fp, -(mem + sizeof(int)), SEEK_CUR);
			op->init_weights(fp);
			return mem;
		}

		template<typename Dtype>
		void operation_convolutiondepthwise_arm<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms, std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			CHECK_EQ(tops.size(), 1);
			CHECK_EQ(this->dilation_h_, 1);
			CHECK_EQ(this->dilation_w_, 1);

			CHECK_EQ(this->input_channel_, this->group_);
			CHECK_EQ(this->output_channel_, this->group_);

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

			if (this->kernel_size_w_ == 3 && this->kernel_size_h_ == 3)
			{
				if (this->stride_w_ == 1 && this->stride_h_ == 1)
					convdw3x3s1_neon(bottom_bordered, tops[0]);
				else if (this->stride_w_ == 2 && this->stride_h_ == 2)
					convdw3x3s2_neon(bottom_bordered, tops[0]);
				else
					NOT_IMPLEMENTED;
			}
			else
			{
				op->forward_cpu(bottoms, tops);
				//NOT_IMPLEMENTED;
			}
			this->suffix_activation_cpu_f32(tops);
		}

		template<typename Dtype>
		void operation_convolutiondepthwise_arm<Dtype>::convdw3x3s1_neon(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top)
		{
			int num = bottom->num();
			int w = bottom->width();
			int h = bottom->height();
			int bottom_cstep = w * h;

			int outw = top->width();
			int outh = top->height();
			int top_cstep = outw * outh;

			const float* kernel = this->weights_f32_[0]->cpu_data();
			const float* bias = nullptr;
			if (this->bias_term_)
				bias = this->weights_f32_[1]->cpu_data();

			for (int num_i = 0; num_i < num; num_i++)
			{
				const float *bottom_data = bottom->cpu_data() + num_i * this->group_ * bottom_cstep;
				float *top_data = top->mutable_cpu_data() + num_i * this->group_ * top_cstep;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
				for (int g = 0; g < this->group_; g++)
				{
					float *out = top_data + (g)* top_cstep;

					const float bias0 = this->bias_term_ ? bias[g] : 0.f;

					const float* kernel0 = kernel + g * 9;

					float* outptr = out;
					float* outptr2 = outptr + outw;

					const float* img0 = bottom_data + (g)* bottom_cstep;

					const float* r0 = img0;
					const float* r1 = img0 + w;
					const float* r2 = img0 + w * 2;
					const float* r3 = img0 + w * 3;

#if __ARM_NEON
					float32x4_t _k012x = vld1q_f32(kernel0);
					float32x4_t _k345x = vld1q_f32(kernel0 + 3);
					float32x4_t _k678x = vld1q_f32(kernel0 + 6);

					_k012x = vsetq_lane_f32(0.f, _k012x, 3);
					_k345x = vsetq_lane_f32(0.f, _k345x, 3);
					_k678x = vsetq_lane_f32(0.f, _k678x, 3);

					float32x4_t _bias0 = vdupq_n_f32(bias0);
#else
					const float* k0 = kernel0;
					const float* k1 = kernel0 + 3;
					const float* k2 = kernel0 + 6;
#endif // __ARM_NEON

					int i = 0;

					for (; i + 1 < outh; i += 2)
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
								"prfm   pldl1keep, [%3, #384]           \n"
								"ld1    {v8.4s, v9.4s, v10.4s}, [%3]    \n"// r0
								"add    %3, %3, #32                     \n"

								"ext    v11.16b, v8.16b, v9.16b, #4     \n"
								"ext    v13.16b, v9.16b, v10.16b, #4    \n"

								"ext    v12.16b, v8.16b, v9.16b, #8     \n"
								"ext    v14.16b, v9.16b, v10.16b, #8    \n"

								"0:                                     \n"

								"and    v4.16b, %17.16b, %17.16b        \n"// v4 = _bias0
								"and    v5.16b, %17.16b, %17.16b        \n"// v5 = _bias0

								"prfm   pldl1keep, [%6, #384]           \n"
								"ld1    {v16.4s, v17.4s, v18.4s}, [%6]  \n"// r3
								"add    %6, %6, #32                     \n"

								"and    v6.16b, %17.16b, %17.16b        \n"// v6 = _bias0
								"and    v7.16b, %17.16b, %17.16b        \n"// v7 = _bias0

								"ext    v15.16b, v16.16b, v17.16b, #4   \n"

								"fmla   v4.4s, v8.4s, %14.s[0]          \n"
								"fmla   v5.4s, v9.4s, %14.s[0]          \n"

								"ext    v20.16b, v17.16b, v18.16b, #4   \n"

								"fmla   v6.4s, v16.4s, %16.s[0]         \n"
								"fmla   v7.4s, v17.4s, %16.s[0]         \n"

								"ext    v19.16b, v16.16b, v17.16b, #8   \n"

								"fmla   v4.4s, v11.4s, %14.s[1]         \n"
								"fmla   v5.4s, v13.4s, %14.s[1]         \n"

								"ext    v21.16b, v17.16b, v18.16b, #8   \n"

								"fmla   v6.4s, v15.4s, %16.s[1]         \n"
								"fmla   v7.4s, v20.4s, %16.s[1]         \n"

								"prfm   pldl1keep, [%4, #384]           \n"
								"ld1    {v22.4s, v23.4s, v24.4s}, [%4]  \n"// r1

								"fmla   v4.4s, v12.4s, %14.s[2]         \n"
								"fmla   v5.4s, v14.4s, %14.s[2]         \n"

								"add    %4, %4, #32                     \n"

								"fmla   v6.4s, v19.4s, %16.s[2]         \n"
								"fmla   v7.4s, v21.4s, %16.s[2]         \n"

								"ext    v25.16b, v22.16b, v23.16b, #4   \n"

								"fmla   v4.4s, v22.4s, %15.s[0]         \n"
								"fmla   v5.4s, v23.4s, %15.s[0]         \n"

								"ext    v27.16b, v23.16b, v24.16b, #4   \n"

								"fmla   v6.4s, v22.4s, %14.s[0]         \n"
								"fmla   v7.4s, v23.4s, %14.s[0]         \n"

								"ext    v26.16b, v22.16b, v23.16b, #8   \n"

								"fmla   v4.4s, v25.4s, %15.s[1]         \n"
								"fmla   v5.4s, v27.4s, %15.s[1]         \n"

								"ext    v28.16b, v23.16b, v24.16b, #8   \n"

								"fmla   v6.4s, v25.4s, %14.s[1]         \n"
								"fmla   v7.4s, v27.4s, %14.s[1]         \n"

								"prfm   pldl1keep, [%5, #384]           \n"
								"ld1    {v8.4s, v9.4s, v10.4s}, [%5]    \n"// r2

								"fmla   v4.4s, v26.4s, %15.s[2]         \n"
								"fmla   v5.4s, v28.4s, %15.s[2]         \n"

								"add    %5, %5, #32                     \n"

								"fmla   v6.4s, v26.4s, %14.s[2]         \n"
								"fmla   v7.4s, v28.4s, %14.s[2]         \n"

								"ext    v11.16b, v8.16b, v9.16b, #4     \n"

								"fmla   v4.4s, v8.4s, %16.s[0]          \n"
								"fmla   v5.4s, v9.4s, %16.s[0]          \n"

								"ext    v13.16b, v9.16b, v10.16b, #4    \n"

								"fmla   v6.4s, v8.4s, %15.s[0]          \n"
								"fmla   v7.4s, v9.4s, %15.s[0]          \n"

								"ext    v12.16b, v8.16b, v9.16b, #8     \n"

								"fmla   v4.4s, v11.4s, %16.s[1]         \n"
								"fmla   v5.4s, v13.4s, %16.s[1]         \n"

								"ext    v14.16b, v9.16b, v10.16b, #8    \n"

								"fmla   v6.4s, v11.4s, %15.s[1]         \n"
								"fmla   v7.4s, v13.4s, %15.s[1]         \n"

								"prfm   pldl1keep, [%3, #384]           \n"
								"ld1    {v8.4s, v9.4s, v10.4s}, [%3]    \n"// r0 next loop

								"fmla   v4.4s, v12.4s, %16.s[2]         \n"
								"fmla   v5.4s, v14.4s, %16.s[2]         \n"

								"add    %3, %3, #32                     \n"
								"ext    v11.16b, v8.16b, v9.16b, #4     \n"

								"fmla   v6.4s, v12.4s, %15.s[2]         \n"
								"fmla   v7.4s, v14.4s, %15.s[2]         \n"

								"ext    v13.16b, v9.16b, v10.16b, #4    \n"
								"ext    v12.16b, v8.16b, v9.16b, #8     \n"

								"st1    {v4.4s, v5.4s}, [%1], #32       \n"

								"ext    v14.16b, v9.16b, v10.16b, #8    \n"

								"subs   %w0, %w0, #1                    \n"

								"st1    {v6.4s, v7.4s}, [%2], #32       \n"

								"bne    0b                              \n"
								"sub    %3, %3, #32                     \n"
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
								"w"(_k012x),      // %14
								"w"(_k345x),      // %15
								"w"(_k678x),      // %16
								"w"(_bias0)       // %17
								: "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28"
								);
						}

						if (remain >= 4)
						{
							remain -= 4;

							asm volatile(
								"prfm   pldl1keep, [%2, #256]           \n"
								"ld1    {v8.4s, v9.4s}, [%2]            \n"// r0
								"add    %2, %2, #16                     \n"

								"and    v4.16b, %15.16b, %15.16b        \n"// v4 = _bias0
								"and    v6.16b, %15.16b, %15.16b        \n"// v6 = _bias0

								"prfm   pldl1keep, [%5, #256]           \n"
								"ld1    {v16.4s, v17.4s}, [%5]          \n"// r3
								"add    %5, %5, #16                     \n"

								"ext    v11.16b, v8.16b, v9.16b, #4     \n"
								"ext    v15.16b, v16.16b, v17.16b, #4   \n"

								"fmla   v4.4s, v8.4s, %12.s[0]          \n"
								"fmla   v6.4s, v16.4s, %14.s[0]         \n"

								"ext    v12.16b, v8.16b, v9.16b, #8     \n"
								"ext    v19.16b, v16.16b, v17.16b, #8   \n"

								"fmla   v4.4s, v11.4s, %12.s[1]         \n"
								"fmla   v6.4s, v15.4s, %14.s[1]         \n"

								"prfm   pldl1keep, [%3, #256]           \n"
								"ld1    {v22.4s, v23.4s}, [%3]          \n"// r1

								"fmla   v4.4s, v12.4s, %12.s[2]         \n"

								"add    %3, %3, #16                     \n"

								"fmla   v6.4s, v19.4s, %14.s[2]         \n"

								"ext    v25.16b, v22.16b, v23.16b, #4   \n"

								"fmla   v4.4s, v22.4s, %13.s[0]         \n"
								"fmla   v6.4s, v22.4s, %12.s[0]         \n"

								"ext    v26.16b, v22.16b, v23.16b, #8   \n"

								"fmla   v4.4s, v25.4s, %13.s[1]         \n"
								"fmla   v6.4s, v25.4s, %12.s[1]         \n"

								"prfm   pldl1keep, [%4, #256]           \n"
								"ld1    {v8.4s, v9.4s}, [%4]            \n"// r2

								"fmla   v4.4s, v26.4s, %13.s[2]         \n"

								"add    %4, %4, #16                     \n"

								"fmla   v6.4s, v26.4s, %12.s[2]         \n"

								"ext    v11.16b, v8.16b, v9.16b, #4     \n"

								"fmla   v4.4s, v8.4s, %14.s[0]          \n"
								"fmla   v6.4s, v8.4s, %13.s[0]          \n"

								"ext    v12.16b, v8.16b, v9.16b, #8     \n"

								"fmla   v4.4s, v11.4s, %14.s[1]         \n"
								"fmla   v6.4s, v11.4s, %13.s[1]         \n"

								"fmla   v4.4s, v12.4s, %14.s[2]         \n"
								"fmla   v6.4s, v12.4s, %13.s[2]         \n"

								"st1    {v4.4s}, [%0], #16              \n"
								"st1    {v6.4s}, [%1], #16              \n"

								: "=r"(outptr),     // %0
								"=r"(outptr2),    // %1
								"=r"(r0),         // %2
								"=r"(r1),         // %3
								"=r"(r2),         // %4
								"=r"(r3)          // %5
								: "0"(outptr),
								"1"(outptr2),
								"2"(r0),
								"3"(r1),
								"4"(r2),
								"5"(r3),
								"w"(_k012x),      // %12
								"w"(_k345x),      // %13
								"w"(_k678x),      // %14
								"w"(_bias0)       // %15
								: "cc", "memory", "v4", "v6", "v8", "v9", "v11", "v12", "v15", "v16", "v17", "v18", "v19", "v22", "v23", "v25", "v26"
								);
						}
#else
						if (nn > 0)
						{
							asm volatile(
								"pld        [%3, #192]          \n"
								"vld1.f32   {d18-d20}, [%3] \n"// r0
								"add        %3, #16             \n"

								"vext.32    q11, q9, q10, #1    \n"
								"vext.32    q12, q9, q10, #2    \n"

								"0:                             \n"

								"vmul.f32   q7, q9, %e14[0]     \n"

								"vand       q13, %q17, %q17     \n"// q13 = _bias0
								"vmul.f32   q6, q11, %e14[1]    \n"
								"vmla.f32   q13, q12, %f14[0]   \n"

								"pld        [%4, #192]          \n"
								"vld1.f32   {d18-d20}, [%4]     \n"// r1
								"add        %4, #16             \n"

								"vmla.f32   q7, q9, %e15[0]     \n"

								"vext.32    q11, q9, q10, #1    \n"
								"vext.32    q12, q9, q10, #2    \n"

								"vmla.f32   q6, q11, %e15[1]    \n"
								"vmla.f32   q13, q12, %f15[0]   \n"

								"vmul.f32   q8, q9, %e14[0]     \n"

								"vand       q15, %q17, %q17     \n"// q15 = _bias0
								"vmul.f32   q14, q11, %e14[1]   \n"
								"vmla.f32   q15, q12, %f14[0]   \n"

								"pld        [%5, #192]          \n"
								"vld1.f32   {d18-d20}, [%5] \n"// r2
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
								"vld1.f32   {d18-d20}, [%3] \n"// r0
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
								"w"(_k012x),      // %14
								"w"(_k345x),      // %15
								"w"(_k678x),      // %16
								"w"(_bias0)       // %17
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

							float32x4_t _sum = vmulq_f32(_r00, _k012x);
							_sum = vmlaq_f32(_sum, _r10, _k345x);
							_sum = vmlaq_f32(_sum, _r20, _k678x);

							float32x4_t _sum2 = vmulq_f32(_r10, _k012x);
							_sum2 = vmlaq_f32(_sum2, _r20, _k345x);
							_sum2 = vmlaq_f32(_sum2, _r30, _k678x);

							_sum = vsetq_lane_f32(bias0, _sum, 3);
							_sum2 = vsetq_lane_f32(bias0, _sum2, 3);
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
							*outptr = mul_add_3x3_native(r0, r1, r2, k0, k1, k2, bias0);
							*outptr2 = mul_add_3x3_native(r1, r2, r3, k0, k1, k2, bias0);
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
								"prfm   pldl1keep, [%2, #384]           \n"
								"ld1    {v8.4s, v9.4s, v10.4s}, [%2]    \n"// r0
								"add    %2, %2, #32                     \n"

								"ext    v12.16b, v8.16b, v9.16b, #4     \n"
								"ext    v14.16b, v9.16b, v10.16b, #4    \n"

								"0:                                     \n"

								"fmul   v6.4s, v8.4s, %10.s[0]          \n"

								"and    v4.16b, %13.16b, %13.16b        \n"// v4 = _bias0

								"fmul   v7.4s, v9.4s, %10.s[0]          \n"

								"and    v5.16b, %13.16b, %13.16b        \n"// v5 = _bias0

								"fmla   v4.4s, v12.4s, %10.s[1]         \n"

								"ext    v13.16b, v8.16b, v9.16b, #8     \n"

								"fmla   v5.4s, v14.4s, %10.s[1]         \n"

								"ext    v15.16b, v9.16b, v10.16b, #8    \n"

								"fmla   v6.4s, v13.4s, %10.s[2]         \n"

								"prfm   pldl1keep, [%3, #384]           \n"
								"ld1    {v16.4s, v17.4s, v18.4s}, [%3]  \n"// r1

								"fmla   v7.4s, v15.4s, %10.s[2]         \n"

								"add    %3, %3, #32                     \n"

								"fmla   v4.4s, v16.4s, %11.s[0]         \n"

								"ext    v20.16b, v16.16b, v17.16b, #4   \n"

								"fmla   v5.4s, v17.4s, %11.s[0]         \n"

								"ext    v22.16b, v17.16b, v18.16b, #4   \n"

								"fmla   v6.4s, v20.4s, %11.s[1]         \n"

								"ext    v21.16b, v16.16b, v17.16b, #8   \n"

								"fmla   v7.4s, v22.4s, %11.s[1]         \n"

								"ext    v23.16b, v17.16b, v18.16b, #8   \n"

								"fmla   v4.4s, v21.4s, %11.s[2]         \n"

								"prfm   pldl1keep, [%4, #384]           \n"
								"ld1    {v24.4s, v25.4s, v26.4s}, [%4]  \n"// r2

								"fmla   v5.4s, v23.4s, %11.s[2]         \n"

								"add    %4, %4, #32                     \n"

								"fmla   v6.4s, v24.4s, %12.s[0]         \n"

								"ext    v12.16b, v24.16b, v25.16b, #4   \n"

								"fmla   v7.4s, v25.4s, %12.s[0]         \n"

								"ext    v14.16b, v25.16b, v26.16b, #4   \n"

								"fmla   v4.4s, v12.4s, %12.s[1]         \n"

								"ext    v13.16b, v24.16b, v25.16b, #8   \n"

								"fmla   v5.4s, v14.4s, %12.s[1]         \n"

								"ext    v15.16b, v25.16b, v26.16b, #8   \n"

								"fmla   v6.4s, v13.4s, %12.s[2]         \n"
								"fmla   v7.4s, v15.4s, %12.s[2]         \n"

								"prfm   pldl1keep, [%2, #384]           \n"
								"ld1    {v8.4s, v9.4s, v10.4s}, [%2]    \n"// r0 next loop

								"fadd   v4.4s, v4.4s, v6.4s             \n"

								"add    %2, %2, #32                     \n"

								"fadd   v5.4s, v5.4s, v7.4s             \n"

								"ext    v12.16b, v8.16b, v9.16b, #4     \n"
								"ext    v14.16b, v9.16b, v10.16b, #4    \n"

								"subs   %w0, %w0, #1                    \n"

								"st1    {v4.4s, v5.4s}, [%1], #32       \n"

								"bne    0b                              \n"
								"sub    %2, %2, #32                     \n"
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
								"w"(_k012x),      // %10
								"w"(_k345x),      // %11
								"w"(_k678x),      // %12
								"w"(_bias0)       // %13
								: "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v20", "v21", "v22", "v23", "v24", "v25", "v26"
								);
						}

						if (remain >= 4)
						{
							remain -= 4;

							asm volatile(
								"prfm   pldl1keep, [%1, #192]           \n"
								"ld1    {v8.4s, v9.4s}, [%1]            \n"// r0
								"add    %1, %1, #16                     \n"

								"and    v4.16b, %11.16b, %11.16b        \n"// v4 = _bias0

								"ext    v12.16b, v8.16b, v9.16b, #4     \n"

								"fmul   v6.4s, v8.4s, %8.s[0]           \n"

								"ext    v13.16b, v8.16b, v9.16b, #8     \n"

								"fmla   v4.4s, v12.4s, %8.s[1]          \n"

								"prfm   pldl1keep, [%2, #192]           \n"
								"ld1    {v16.4s, v17.4s}, [%2]          \n"// r1
								"add    %2, %2, #16                     \n"

								"fmla   v6.4s, v13.4s, %8.s[2]          \n"

								"ext    v20.16b, v16.16b, v17.16b, #4   \n"

								"fmla   v4.4s, v16.4s, %9.s[0]          \n"

								"ext    v21.16b, v16.16b, v17.16b, #8   \n"

								"fmla   v6.4s, v20.4s, %9.s[1]          \n"

								"prfm   pldl1keep, [%3, #192]           \n"
								"ld1    {v24.4s, v25.4s}, [%3]          \n"// r2
								"add    %3, %3, #16                     \n"

								"fmla   v4.4s, v21.4s, %9.s[2]          \n"

								"ext    v12.16b, v24.16b, v25.16b, #4   \n"

								"fmla   v6.4s, v24.4s, %10.s[0]         \n"

								"ext    v13.16b, v24.16b, v25.16b, #8   \n"

								"fmla   v4.4s, v12.4s, %10.s[1]         \n"

								"fmla   v6.4s, v13.4s, %10.s[2]         \n"

								"fadd   v4.4s, v4.4s, v6.4s             \n"

								"st1    {v4.4s}, [%0], #16              \n"

								: "=r"(outptr),     // %0
								"=r"(r0),         // %1
								"=r"(r1),         // %2
								"=r"(r2)          // %3
								: "0"(outptr),
								"1"(r0),
								"2"(r1),
								"3"(r2),
								"w"(_k012x),      // %8
								"w"(_k345x),      // %9
								"w"(_k678x),      // %10
								"w"(_bias0)       // %11
								: "cc", "memory", "v4", "v6", "v8", "v9", "v12", "v13", "v16", "v17", "v20", "v21", "v24", "v25"
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

								"vmul.f32   q7, q8, %e10[0]     \n"

								"vand       q14, %q13, %q13     \n"// q14 = _bias0
								"vmul.f32   q13, q10, %e10[1]   \n"
								"vmla.f32   q14, q11, %f10[0]   \n"

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
								"w"(_k012x),      // %10
								"w"(_k345x),      // %11
								"w"(_k678x),      // %12
								"w"(_bias0)       // %13
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

							float32x4_t _sum = vmulq_f32(_r00, _k012x);
							_sum = vmlaq_f32(_sum, _r10, _k345x);
							_sum = vmlaq_f32(_sum, _r20, _k678x);

							_sum = vsetq_lane_f32(bias0, _sum, 3);
#if __aarch64__
							*outptr = vaddvq_f32(_sum);
#else
							float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
							_ss = vpadd_f32(_ss, _ss);

							*outptr = vget_lane_f32(_ss, 0);
#endif // __aarch64__
#else
							*outptr = mul_add_3x3_native(r0, r1, r2, k0, k1, k2, bias0);
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
				}
			}
		}

		template<typename Dtype>
		void operation_convolutiondepthwise_arm<Dtype>::convdw3x3s2_neon(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top)
		{
			int num = bottom->num();
			int w = bottom->width();
			int h = bottom->height();
			int bottom_cstep = w * h;

			int outw = top->width();
			int outh = top->height();
			int top_cstep = outw * outh;

			const int tailstep = w - 2 * outw + w;

			const float* kernel = this->weights_f32_[0]->cpu_data();

			const float* bias = nullptr;
			if (this->bias_term_)
				bias = this->weights_f32_[1]->cpu_data();

			for (int num_i = 0; num_i < num; num_i++)
			{
				const float *bottom_data = bottom->cpu_data() + num_i * this->group_ * bottom_cstep;
				float *top_data = top->mutable_cpu_data() + num_i * this->group_ * top_cstep;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
				for (int g = 0; g < this->group_; g++)
				{
					float *out = top_data + (g)* top_cstep;

					const float bias0 = bias ? bias[g] : 0.f;

					const float* kernel0 = kernel + g * 9;

					float* outptr = out;

					const float* img0 = bottom_data + (g)* bottom_cstep;

					const float* r0 = img0;
					const float* r1 = img0 + w;
					const float* r2 = img0 + w * 2;

#if __ARM_NEON
					float32x4_t _k012x = vld1q_f32(kernel0);
					float32x4_t _k345x = vld1q_f32(kernel0 + 3);
					float32x4_t _k678x = vld1q_f32(kernel0 + 6);

					_k012x = vsetq_lane_f32(0.f, _k012x, 3);
					_k345x = vsetq_lane_f32(0.f, _k345x, 3);
					_k678x = vsetq_lane_f32(0.f, _k678x, 3);

					float32x4_t _bias0 = vdupq_n_f32(bias0);
#else
					const float* k0 = kernel0;
					const float* k1 = kernel0 + 3;
					const float* k2 = kernel0 + 6;
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

								"and        v11.16b, %13.16b, %13.16b      \n" // v11 = _bias0

								"0:                                        \n"
								"fmul       v0.4s,  v2.4s, %10.s[0]        \n"
								"fmul       v10.4s, v3.4s, %10.s[1]        \n"

								"prfm       pldl1keep, [%2, #256]          \n"
								"ld2        {v8.4s, v9.4s}, [%2]           \n"
								"ext        v1.16b, v2.16b, v8.16b, #4     \n"

								"fmla       v11.4s, v1.4s, %10.s[2]        \n"

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

								"and        v11.16b, %13.16b, %13.16b      \n" // v11 = _bias0

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
								"w"(_k012x),  // %10
								"w"(_k345x),  // %11
								"w"(_k678x),  // %12
								"w"(_bias0)   // %13
								: "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
								);
						}
#else
						if (nn > 0)
						{
							asm volatile(
								"pld        [%2, #256]          \n"
								"vld2.f32   {d4-d7}, [%2]!      \n"

								"vand       q11, %q13, %q13     \n"

								"0:                             \n"
								"vmul.f32   q0, q2, %e10[0]     \n"
								"vmul.f32   q10, q3, %e10[1]    \n"

								"pld        [%2, #128]          \n"
								"vld2.f32   {d16-d17}, [%2]     \n"
								"vext.32    q1, q2, q8, #1      \n"

								"vmla.f32   q11, q1, %f10[0]    \n"

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

								"vand       q11, %q13, %q13     \n"

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
								"w"(_k012x),  // %10
								"w"(_k345x),  // %11
								"w"(_k678x),  // %12
								"w"(_bias0)   // %13
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

							float32x4_t _sum = vmulq_f32(_r00, _k012x);
							_sum = vmlaq_f32(_sum, _r10, _k345x);
							_sum = vmlaq_f32(_sum, _r20, _k678x);

							_sum = vsetq_lane_f32(bias0, _sum, 3);
#if __aarch64__
							*outptr = vaddvq_f32(_sum);
#else
							float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
							_ss = vpadd_f32(_ss, _ss);

							*outptr = vget_lane_f32(_ss, 0);
#endif // __aarch64__
#else
							*outptr = mul_add_3x3_native(r0, r1, r2, k0, k1, k2, bias0);
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

				}
			}
		}

		template<typename Dtype>
		void operation_convolutiondepthwise_arm<Dtype>::forward_cpu_i8(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			CHECK_EQ(tops.size(), 1);
			CHECK_EQ(this->dilation_h_, 1);
			CHECK_EQ(this->dilation_w_, 1);

			CHECK_EQ(this->input_channel_, this->group_);
			CHECK_EQ(this->output_channel_, this->group_);

			int n = bottoms[0]->num();
			int w = bottoms[0]->width();
			int h = bottoms[0]->height();

			std::shared_ptr<memory::tensor<int8_t>> bottom_unbordered(new memory::tensor<int8_t>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order()));

			const float* bottom_data = bottoms[0]->cpu_data();
			int8_t *bottom_unbordered_data = bottom_unbordered->mutable_cpu_data();
			this->quantize_float32_to_int8(bottoms[0], bottom_unbordered);

			std::shared_ptr<memory::tensor<int8_t> > bottom_bordered = bottom_unbordered;

			if (this->pad_left_ > 0 || this->pad_top_ > 0 || this->pad_right_ > 0 || this->pad_bottom_ > 0)
			{
				make_border(bottom_unbordered, bottom_bordered, this->pad_top_, this->pad_bottom_, this->pad_left_, this->pad_right_);
				w = bottom_bordered->width();
				h = bottom_bordered->height();
			}
			int outw = (w - this->kernel_size_w_) / this->stride_w_ + 1;
			int outh = (h - this->kernel_size_h_) / this->stride_h_ + 1;

			memory::orderType order = bottom_bordered->order();
			if (!((order == memory::NCHW) || (order == memory::NHWC)))
			{
				NOT_IMPLEMENTED;
			}

			if (order == memory::NHWC)
			{
				bottom_bordered->convert_order();
			}

			std::shared_ptr<memory::tensor<int>> top(new memory::tensor<int>(std::vector<int>{n, this->output_channel_, outh, outw }, this->params_.device_, memory::NCHW));
			tops[0].reset(new memory::tensor<float>(top->data_shape(), this->params_.device_, top->order()));

			if (this->kernel_size_w_ == 3 && this->kernel_size_h_ == 3 && this->stride_w_ == 1 && this->stride_h_ == 1)
			{
				convdw3x3s1_int8_neon(bottom_bordered, top);
			}
			else if (this->kernel_size_w_ == 3 && this->kernel_size_h_ == 3 && this->stride_w_ == 2 && this->stride_h_ == 2)
			{
				convdw3x3s2_int8_neon(bottom_bordered, top);
			}
			else
			{
				op->forward_cpu(bottoms, tops);
				return;

				//NOT_IMPLEMENTED;
			}

			this->dequantize_int32_to_float32(top, tops[0]);
			this->suffix_activation_cpu_f32(tops);
		}

		template<typename Dtype>
		void operation_convolutiondepthwise_arm<Dtype>::convdw3x3s1_int8_neon(const std::shared_ptr<memory::tensor<int8_t>>& bottom, std::shared_ptr<memory::tensor<int>>& top)
		{
			int num = bottom->num();
			int w = bottom->width();
			int h = bottom->height();
			int bottom_cstep = w * h;

			int outw = top->width();
			int outh = top->height();
			int outch = top->channels();
			int top_cstep = outw * outh;

			const int8_t* _kernel = this->weights_i8_[0]->cpu_data();

			for (size_t n = 0; n < num; n++)
			{
				const int8_t* bottom_data = bottom->cpu_data() + n * bottom->count(1,4);
				int* top_data = top->mutable_cpu_data() + n * top->count(1, 4);

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
				for (int p = 0; p < outch; p++)
				{
					int* out = top_data + p * top_cstep;

					const signed char* kernel = _kernel + p * 9;

					int* outptr0 = out;
					int* outptr0n = outptr0 + outw;

					const signed char* img0 = bottom_data + p * bottom_cstep;

					const signed char* r0 = img0;
					const signed char* r1 = img0 + w;
					const signed char* r2 = img0 + w * 2;
					const signed char* r3 = img0 + w * 3;

					int i = 0;

#if __ARM_NEON
					int8x16_t _k0123456789x = vld1q_s8(kernel);
					int16x8_t _k_s16 = vmovl_s8(vget_low_s8(_k0123456789x));
					int16x8_t _kn_s16 = vmovl_s8(vget_high_s8(_k0123456789x));

					int16x4_t _k0123 = vget_low_s16(_k_s16);
					int16x4_t _k4567 = vget_high_s16(_k_s16);
					int16x4_t _k8xxx = vget_low_s16(_kn_s16);
#endif // __ARM_NEON

					for (; i + 1 < outh; i += 2)
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
								"ld1    {v4.8b, v5.8b}, [%3]          \n"
								"ld1    {v6.8b, v7.8b}, [%4]          \n"
								"ld1    {v8.8b, v9.8b}, [%5]          \n"
								"ld1    {v10.8b, v11.8b}, [%6]        \n"
								"add    %3, %3, #8                    \n"
								"add    %4, %4, #8                    \n"
								"add    %5, %5, #8                    \n"
								"add    %6, %6, #8                    \n"

								"ext    v12.8b, v4.8b, v5.8b, #1      \n"
								"ext    v13.8b, v4.8b, v5.8b, #2      \n"
								"ext    v14.8b, v6.8b, v7.8b, #1      \n"
								"ext    v15.8b, v6.8b, v7.8b, #2      \n"
								"ext    v16.8b, v8.8b, v9.8b, #1      \n"
								"ext    v17.8b, v8.8b, v9.8b, #2      \n"
								"ext    v18.8b, v10.8b, v11.8b, #1    \n"
								"ext    v19.8b, v10.8b, v11.8b, #2    \n"

								"sshll  v4.8h, v4.8b, #0              \n" // r00
								"sshll  v12.8h, v12.8b, #0            \n" // r01
								"sshll  v13.8h, v13.8b, #0            \n" // r02
								"sshll  v6.8h, v6.8b, #0              \n" // r10
								"sshll  v14.8h, v14.8b, #0            \n" // r11
								"sshll  v15.8h, v15.8b, #0            \n" // r12
								"sshll  v8.8h, v8.8b, #0              \n" // r20
								"sshll  v16.8h, v16.8b, #0            \n" // r21
								"sshll  v17.8h, v17.8b, #0            \n" // r22
								"sshll  v10.8h, v10.8b, #0            \n" // r30
								"sshll  v18.8h, v18.8b, #0            \n" // r31
								"sshll  v19.8h, v19.8b, #0            \n" // r32

								// r0
								"smull  v20.4s, v4.4h, %14.h[0]       \n" // (r00 - r07) * k00
								"smull2  v21.4s, v4.8h, %14.h[0]      \n"
								"smull  v22.4s, v12.4h, %14.h[1]      \n" // (r01 - r08) * k01
								"smull2  v23.4s, v12.8h, %14.h[1]     \n"
								"smull  v24.4s, v13.4h, %14.h[2]      \n" // (r02 - r09) * k02
								"smull2  v25.4s, v13.8h, %14.h[2]     \n"

								// r1
								"smull  v26.4s, v6.4h, %14.h[0]       \n" // (r10 - r17) * k00
								"smull2  v27.4s, v6.8h, %14.h[0]      \n"
								"smull  v28.4s, v14.4h, %14.h[1]      \n" // (r11 - r18) * k01
								"smull2  v29.4s, v14.8h, %14.h[1]     \n"
								"smull  v30.4s, v15.4h, %14.h[2]      \n" // (r12 - r19) * k02
								"smull2  v31.4s, v15.8h, %14.h[2]     \n"

								"smlal  v20.4s, v6.4h, %14.h[3]       \n" // (r10 - r17) * k03
								"smlal2  v21.4s, v6.8h, %14.h[3]      \n"
								"smlal  v22.4s, v14.4h, %15.h[0]      \n" // (r11 - r18) * k04
								"smlal2  v23.4s, v14.8h, %15.h[0]     \n"
								"smlal  v24.4s, v15.4h, %15.h[1]      \n" // (r12 - r19) * k05
								"smlal2  v25.4s, v15.8h, %15.h[1]     \n"

								// r2
								"smlal  v26.4s, v8.4h, %14.h[3]       \n" // (r20 - r27) * k03
								"smlal2  v27.4s, v8.8h, %14.h[3]      \n"
								"smlal  v28.4s, v16.4h, %15.h[0]      \n" // (r21 - r28) * k04
								"smlal2  v29.4s, v16.8h, %15.h[0]     \n"
								"smlal  v30.4s, v17.4h, %15.h[1]      \n" // (r22 - r29) * k05
								"smlal2  v31.4s, v17.8h, %15.h[1]     \n"

								"smlal  v20.4s, v8.4h, %15.h[2]       \n" // (r20 - r27) * k06
								"smlal2  v21.4s, v8.8h, %15.h[2]      \n"
								"smlal  v22.4s, v16.4h, %15.h[3]      \n" // (r21 - r28) * k07
								"smlal2  v23.4s, v16.8h, %15.h[3]     \n"
								"smlal  v24.4s, v17.4h, %16.h[0]      \n" // (r22 - r29) * k08
								"smlal2  v25.4s, v17.8h, %16.h[0]     \n"

								// r3
								"smlal  v26.4s, v10.4h, %15.h[2]      \n" // (r30 - r37) * k06
								"smlal2  v27.4s, v10.8h, %15.h[2]     \n"
								"smlal  v28.4s, v18.4h, %15.h[3]      \n" // (r31 - r38) * k07
								"smlal2  v29.4s, v18.8h, %15.h[3]     \n"
								"smlal  v30.4s, v19.4h, %16.h[0]      \n" // (r32 - r39) * k08
								"smlal2  v31.4s, v19.8h, %16.h[0]     \n"

								// add and save
								"add    v20.4s, v20.4s, v22.4s        \n"
								"add    v21.4s, v21.4s, v23.4s        \n"
								"add    v26.4s, v26.4s, v28.4s        \n"
								"add    v27.4s, v27.4s, v29.4s        \n"
								"add    v20.4s, v20.4s, v24.4s        \n"
								"add    v21.4s, v21.4s, v25.4s        \n"
								"add    v26.4s, v26.4s, v30.4s        \n"
								"add    v27.4s, v27.4s, v31.4s        \n"

								"st1    {v20.4s, v21.4s}, [%1], #32   \n"
								"st1    {v26.4s, v27.4s}, [%2], #32   \n"

								"subs   %w0, %w0, #1                  \n"
								"bne    0b                            \n"

								: "=r"(nn),       // %0
								"=r"(outptr0),  // %1
								"=r"(outptr0n), // %2
								"=r"(r0),       // %3
								"=r"(r1),       // %4
								"=r"(r2),       // %5
								"=r"(r3)        // %6
								: "0"(nn),
								"1"(outptr0),
								"2"(outptr0n),
								"3"(r0),
								"4"(r1),
								"5"(r2),
								"6"(r3),
								"w"(_k0123), // %14
								"w"(_k4567), // %15
								"w"(_k8xxx)  // %16
								: "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
						}
#else
						if (nn > 0)
						{
							asm volatile(
								"0:                              \n"
								// r0
								"vld1.s8    {d30-d31}, [%3]      \n" // r0
								"add    %3, %3, #8               \n"

								"vext.s8    d10, d30, d31, #1    \n"
								"vext.s8    d12, d30, d31, #2    \n"

								"vmovl.s8    q15, d30            \n" // r00
								"vmovl.s8    q5, d10             \n" // r01
								"vmovl.s8    q6, d12             \n" // r02
								// sum0
								"vmull.s16  q7, d30, %P14[0]     \n" // (r00 - r07) * k00
								"vmull.s16  q8, d31, %P14[0]     \n"
								"vmull.s16  q9, d10, %P14[1]     \n" // (r01 - r08) * k01
								"vmull.s16  q10, d11, %P14[1]    \n"
								"vmlal.s16  q7, d12, %P14[2]     \n" // (r02 - r09) * k02
								"vmlal.s16  q8, d13, %P14[2]     \n"

								// r1
								"vld1.s8    {d30-d31}, [%4]      \n" // r1
								"add    %4, %4, #8               \n"

								"vext.s8    d10, d30, d31, #1    \n"
								"vext.s8    d12, d30, d31, #2    \n"

								"vmovl.s8    q15, d30            \n" // r10
								"vmovl.s8    q5, d10             \n" // r11
								"vmovl.s8    q6, d12             \n" // r12
								// sum0
								"vmlal.s16  q7, d30, %P14[3]     \n" // (r10 - r17) * k03
								"vmlal.s16  q8, d31, %P14[3]     \n"
								"vmlal.s16  q9, d10, %P15[0]     \n" // (r11 - r18) * k04
								"vmlal.s16  q10, d11, %P15[0]    \n"
								"vmlal.s16  q7, d12, %P15[1]     \n" // (r12 - r19) * k05
								"vmlal.s16  q8, d13, %P15[1]     \n"
								// sum1
								"vmull.s16  q11, d30, %P14[0]    \n" // (r10 - r17) * k00
								"vmull.s16  q12, d31, %P14[0]    \n"
								"vmull.s16  q13, d10, %P14[1]    \n" // (r11 - r18) * k01
								"vmull.s16  q14, d11, %P14[1]    \n"
								"vmlal.s16  q11, d12, %P14[2]    \n" // (r12 - r19) * k02
								"vmlal.s16  q12, d13, %P14[2]    \n"

								// r2
								"vld1.s8    {d30-d31}, [%5]      \n" // r2
								"add    %5, %5, #8               \n"

								"vext.s8    d10, d30, d31, #1    \n"
								"vext.s8    d12, d30, d31, #2    \n"

								"vmovl.s8    q15, d30            \n" // r20
								"vmovl.s8    q5, d10             \n" // r21
								"vmovl.s8    q6, d12             \n" // r22

								// sum0
								"vmlal.s16  q7, d30, %P15[2]     \n" // (r20 - r27) * k06
								"vmlal.s16  q8, d31, %P15[2]     \n"
								"vmlal.s16  q9, d10, %P15[3]     \n" // (r21 - r28) * k07
								"vmlal.s16  q10, d11, %P15[3]    \n"
								"vmlal.s16  q7, d12, %P16[0]     \n" // (r22 - r29) * k08
								"vmlal.s16  q8, d13, %P16[0]     \n"
								// sum1
								"vmlal.s16  q11, d30, %P14[3]    \n" // (r20 - r27) * k03
								"vmlal.s16  q12, d31, %P14[3]    \n"
								"vmlal.s16  q13, d10, %P15[0]    \n" // (r21 - r28) * k04
								"vmlal.s16  q14, d11, %P15[0]    \n"
								"vmlal.s16  q11, d12, %P15[1]    \n" // (r22 - r29) * k05
								"vmlal.s16  q12, d13, %P15[1]    \n"

								// r3
								"vld1.s8    {d30-d31}, [%6]      \n" // r3
								"add    %6, %6, #8               \n"

								"vext.s8    d10, d30, d31, #1    \n"
								"vext.s8    d12, d30, d31, #2    \n"

								"vmovl.s8    q15, d30            \n" // r30
								"vmovl.s8    q5, d10             \n" // r31
								"vmovl.s8    q6, d12             \n" // r32

								// sum1
								"vmlal.s16  q11, d30, %P15[2]    \n" // (r30 - r37) * k06
								"vmlal.s16  q12, d31, %P15[2]    \n"
								"vmlal.s16  q13, d10, %P15[3]    \n" // (r31 - r38) * k07
								"vmlal.s16  q14, d11, %P15[3]    \n"
								"vmlal.s16  q11, d12, %P16[0]    \n" // (r32 - r39) * k08
								"vmlal.s16  q12, d13, %P16[0]    \n"

								"subs   %0, %0, #1               \n"

								// add and save
								"vadd.s32    q7, q7, q9          \n"
								"vadd.s32    q8, q8, q10         \n"
								"vadd.s32    q11, q11, q13       \n"
								"vadd.s32    q12, q12, q14       \n"

								"vst1.s32    {d14-d17}, [%1]!    \n"
								"vst1.s32    {d22-d25}, [%2]!    \n"

								"bne    0b                       \n"

								: "=r"(nn),       // %0
								"=r"(outptr0),  // %1
								"=r"(outptr0n), // %2
								"=r"(r0),       // %3
								"=r"(r1),       // %4
								"=r"(r2),       // %5
								"=r"(r3)        // %6
								: "0"(nn),
								"1"(outptr0),
								"2"(outptr0n),
								"3"(r0),
								"4"(r1),
								"5"(r2),
								"6"(r3),
								"w"(_k0123), // %14
								"w"(_k4567), // %15
								"w"(_k8xxx)  // %16
								: "cc", "memory", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
				}
#endif // __aarch64__
#endif // __ARM_NEON
						for (; remain > 0; remain--)
						{
							// TODO NEON
							int sum0 = 0;
							int sum0n = 0;

							sum0 += (int)r0[0] * kernel[0];
							sum0 += (int)r0[1] * kernel[1];
							sum0 += (int)r0[2] * kernel[2];
							sum0 += (int)r1[0] * kernel[3];
							sum0 += (int)r1[1] * kernel[4];
							sum0 += (int)r1[2] * kernel[5];
							sum0 += (int)r2[0] * kernel[6];
							sum0 += (int)r2[1] * kernel[7];
							sum0 += (int)r2[2] * kernel[8];

							sum0n += (int)r1[0] * kernel[0];
							sum0n += (int)r1[1] * kernel[1];
							sum0n += (int)r1[2] * kernel[2];
							sum0n += (int)r2[0] * kernel[3];
							sum0n += (int)r2[1] * kernel[4];
							sum0n += (int)r2[2] * kernel[5];
							sum0n += (int)r3[0] * kernel[6];
							sum0n += (int)r3[1] * kernel[7];
							sum0n += (int)r3[2] * kernel[8];

							*outptr0 = sum0;
							*outptr0n = sum0n;

							r0++;
							r1++;
							r2++;
							r3++;
							outptr0++;
							outptr0n++;
						}

						r0 += 2 + w;
						r1 += 2 + w;
						r2 += 2 + w;
						r3 += 2 + w;

						outptr0 += outw;
						outptr0n += outw;
			}

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
								"ld1    {v4.8b, v5.8b}, [%2]          \n"
								"ld1    {v6.8b, v7.8b}, [%3]          \n"
								"ld1    {v8.8b, v9.8b}, [%4]          \n"
								"add    %2, %2, #8                    \n"
								"add    %3, %3, #8                    \n"
								"add    %4, %4, #8                    \n"

								"ext    v12.8b, v4.8b, v5.8b, #1      \n"
								"ext    v13.8b, v4.8b, v5.8b, #2      \n"
								"ext    v14.8b, v6.8b, v7.8b, #1      \n"
								"ext    v15.8b, v6.8b, v7.8b, #2      \n"
								"ext    v16.8b, v8.8b, v9.8b, #1      \n"
								"ext    v17.8b, v8.8b, v9.8b, #2      \n"

								"sshll  v4.8h, v4.8b, #0              \n" // r00
								"sshll  v12.8h, v12.8b, #0            \n" // r01
								"sshll  v13.8h, v13.8b, #0            \n" // r02
								"sshll  v6.8h, v6.8b, #0              \n" // r10
								"sshll  v14.8h, v14.8b, #0            \n" // r11
								"sshll  v15.8h, v15.8b, #0            \n" // r12
								"sshll  v8.8h, v8.8b, #0              \n" // r20
								"sshll  v16.8h, v16.8b, #0            \n" // r21
								"sshll  v17.8h, v17.8b, #0            \n" // r22

								// r0
								"smull  v20.4s, v4.4h, %10.h[0]       \n" // (r00 - r07) * k00
								"smull2  v21.4s, v4.8h, %10.h[0]      \n"
								"smull  v22.4s, v12.4h, %10.h[1]      \n" // (r01 - r08) * k01
								"smull2  v23.4s, v12.8h, %10.h[1]     \n"
								"smull  v24.4s, v13.4h, %10.h[2]      \n" // (r02 - r09) * k02
								"smull2  v25.4s, v13.8h, %10.h[2]     \n"

								// r1
								"smlal  v20.4s, v6.4h, %10.h[3]       \n" // (r10 - r17) * k03
								"smlal2  v21.4s, v6.8h, %10.h[3]      \n"
								"smlal  v22.4s, v14.4h, %11.h[0]      \n" // (r11 - r18) * k04
								"smlal2  v23.4s, v14.8h, %11.h[0]     \n"
								"smlal  v24.4s, v15.4h, %11.h[1]      \n" // (r12 - r19) * k05
								"smlal2  v25.4s, v15.8h, %11.h[1]     \n"

								// r2
								"smlal  v20.4s, v8.4h, %11.h[2]       \n" // (r20 - r27) * k06
								"smlal2  v21.4s, v8.8h, %11.h[2]      \n"
								"smlal  v22.4s, v16.4h, %11.h[3]      \n" // (r21 - r28) * k07
								"smlal2  v23.4s, v16.8h, %11.h[3]     \n"
								"smlal  v24.4s, v17.4h, %12.h[0]      \n" // (r22 - r29) * k08
								"smlal2  v25.4s, v17.8h, %12.h[0]     \n"

								// add and save
								"add    v20.4s, v20.4s, v22.4s        \n"
								"add    v21.4s, v21.4s, v23.4s        \n"
								"add    v20.4s, v20.4s, v24.4s        \n"
								"add    v21.4s, v21.4s, v25.4s        \n"

								"st1    {v20.4s, v21.4s}, [%1], #32   \n"

								"subs   %w0, %w0, #1                  \n"
								"bne    0b                            \n"

								: "=r"(nn),      // %0
								"=r"(outptr0), // %1
								"=r"(r0),      // %2
								"=r"(r1),      // %3
								"=r"(r2)       // %4
								: "0"(nn),
								"1"(outptr0),
								"2"(r0),
								"3"(r1),
								"4"(r2),
								"w"(_k0123), // %10
								"w"(_k4567), // %11
								"w"(_k8xxx)  // %12
								: "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25");
						}
#else
						if (nn > 0)
						{
							asm volatile(
								"0:                              \n"
								// r0
								"vld1.s8    {d30-d31}, [%2]        \n" // r0
								"add    %2, %2, #8               \n"

								"vext.s8    d10, d30, d31, #1      \n"
								"vext.s8    d12, d30, d31, #2      \n"

								"vmovl.s8    q15, d30              \n" // r00
								"vmovl.s8    q5, d10             \n"   // r01
								"vmovl.s8    q6, d12             \n"   // r02
								// sum0
								"vmull.s16  q7, d30, %P10[0]      \n" // (r00 - r07) * k00
								"vmull.s16  q8, d31, %P10[0]      \n"
								"vmull.s16  q9, d10, %P10[1]     \n" // (r01 - r08) * k01
								"vmull.s16  q10, d11, %P10[1]    \n"
								"vmlal.s16  q7, d12, %P10[2]     \n" // (r02 - r09) * k02
								"vmlal.s16  q8, d13, %P10[2]     \n"

								// r1
								"vld1.s8    {d30-d31}, [%3]        \n" // r1
								"add    %3, %3, #8               \n"

								"vext.s8    d10, d30, d31, #1      \n"
								"vext.s8    d12, d30, d31, #2      \n"

								"vmovl.s8    q15, d30              \n" // r10
								"vmovl.s8    q5, d10             \n"   // r11
								"vmovl.s8    q6, d12             \n"   // r12
								// sum0
								"vmlal.s16  q7, d30, %P10[3]      \n" // (r10 - r17) * k03
								"vmlal.s16  q8, d31, %P10[3]      \n"
								"vmlal.s16  q9, d10, %P11[0]     \n" // (r11 - r18) * k04
								"vmlal.s16  q10, d11, %P11[0]    \n"
								"vmlal.s16  q7, d12, %P11[1]     \n" // (r12 - r19) * k05
								"vmlal.s16  q8, d13, %P11[1]     \n"

								// r2
								"vld1.s8    {d30-d31}, [%4]        \n" // r2
								"add    %4, %4, #8               \n"

								"vext.s8    d10, d30, d31, #1      \n"
								"vext.s8    d12, d30, d31, #2      \n"

								"vmovl.s8    q15, d30              \n" // r20
								"vmovl.s8    q5, d10             \n"   // r21
								"vmovl.s8    q6, d12             \n"   // r22

								// sum0
								"vmlal.s16  q7, d30, %P11[2]      \n" // (r20 - r27) * k06
								"vmlal.s16  q8, d31, %P11[2]      \n"
								"vmlal.s16  q9, d10, %P11[3]     \n" // (r21 - r28) * k07
								"vmlal.s16  q10, d11, %P11[3]    \n"
								"vmlal.s16  q7, d12, %P12[0]     \n" // (r22 - r29) * k08
								"vmlal.s16  q8, d13, %P12[0]     \n"

								"subs   %0, %0, #1               \n"

								// add and save
								"vadd.s32    q7, q7, q9          \n"
								"vadd.s32    q8, q8, q10         \n"

								"vst1.s32    {d14-d17}, [%1]!    \n"

								"bne    0b                       \n"

								: "=r"(nn),      // %0
								"=r"(outptr0), // %1
								"=r"(r0),      // %2
								"=r"(r1),      // %3
								"=r"(r2)       // %4
								: "0"(nn),
								"1"(outptr0),
								"2"(r0),
								"3"(r1),
								"4"(r2),
								"w"(_k0123), // %10
								"w"(_k4567), // %11
								"w"(_k8xxx)  // %12
								: "cc", "memory", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
		}
#endif // __aarch64__
#endif // __ARM_NEON
						for (; remain > 0; remain--)
						{
							int sum = 0;

							sum += (int)r0[0] * kernel[0];
							sum += (int)r0[1] * kernel[1];
							sum += (int)r0[2] * kernel[2];
							sum += (int)r1[0] * kernel[3];
							sum += (int)r1[1] * kernel[4];
							sum += (int)r1[2] * kernel[5];
							sum += (int)r2[0] * kernel[6];
							sum += (int)r2[1] * kernel[7];
							sum += (int)r2[2] * kernel[8];

							*outptr0 = sum;

							r0++;
							r1++;
							r2++;
							outptr0++;
						}

						r0 += 2;
						r1 += 2;
						r2 += 2;
					}
				}
			}
		}

		template<typename Dtype>
		void operation_convolutiondepthwise_arm<Dtype>::convdw3x3s2_int8_neon(const std::shared_ptr<memory::tensor<int8_t>>& bottom, std::shared_ptr<memory::tensor<int>>& top)
		{
			int num = bottom->num();
			int w = bottom->width();
			int h = bottom->height();
			int bottom_cstep = w * h;

			int outw = top->width();
			int outh = top->height();
			int outch = top->channels();
			int top_cstep = outw * outh;

			const int tailstep = w - 2 * outw + w;
			
			const int8_t* _kernel = this->weights_i8_[0]->cpu_data();
			for (size_t n = 0; n < num; n++)
			{
				const int8_t* bottom_data = bottom->cpu_data() + n * bottom->count(1, 4);
				int* top_data = top->mutable_cpu_data() + n * top->count(1, 4);

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
				for (int p = 0; p < outch; p++)
				{
					int* out = top_data + p * top_cstep;

					const signed char* kernel = _kernel + p * 9;

					int* outptr = out;

					const signed char* img = bottom_data + p * bottom_cstep;

					const signed char* r0 = img;
					const signed char* r1 = img + w;
					const signed char* r2 = img + w * 2;

					int i = 0;
#if __ARM_NEON
					int8x16_t _k0123456789x = vld1q_s8(kernel);
					int16x8_t _k_s16 = vmovl_s8(vget_low_s8(_k0123456789x));
					int16x8_t _kn_s16 = vmovl_s8(vget_high_s8(_k0123456789x));

					int16x4_t _k0123 = vget_low_s16(_k_s16);
					int16x4_t _k4567 = vget_high_s16(_k_s16);
					int16x4_t _k8xxx = vget_low_s16(_kn_s16);
#endif // __ARM_NEON
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
								"ld2    {v4.8b, v5.8b}, [%2], #16     \n"
								"ld2    {v6.8b, v7.8b}, [%2]          \n"
								"ld2    {v8.8b, v9.8b}, [%3], #16     \n"
								"ld2    {v10.8b, v11.8b}, [%3]        \n"
								"ld2    {v12.8b, v13.8b}, [%4], #16   \n"
								"ld2    {v14.8b, v15.8b}, [%4]        \n"

								"ext    v6.8b, v4.8b, v6.8b, #1       \n"
								"ext    v10.8b, v8.8b, v10.8b, #1     \n"
								"ext    v14.8b, v12.8b, v14.8b, #1    \n"

								"sshll  v4.8h, v4.8b, #0              \n" // r00
								"sshll  v5.8h, v5.8b, #0              \n" // r01
								"sshll  v6.8h, v6.8b, #0              \n" // r02
								"sshll  v8.8h, v8.8b, #0              \n" // r10
								"sshll  v9.8h, v9.8b, #0              \n" // r11
								"sshll  v10.8h, v10.8b, #0            \n" // r12
								"sshll  v12.8h, v12.8b, #0            \n" // r20
								"sshll  v13.8h, v13.8b, #0            \n" // r21
								"sshll  v14.8h, v14.8b, #0            \n" // r22

								// r0
								"smull  v20.4s, v4.4h, %10.h[0]       \n" // (r00 - r07) * k00
								"smull2  v21.4s, v4.8h, %10.h[0]      \n"
								"smull  v22.4s, v5.4h, %10.h[1]       \n" // (r01 - r08) * k01
								"smull2  v23.4s, v5.8h, %10.h[1]      \n"
								"smull  v24.4s, v6.4h, %10.h[2]       \n" // (r02 - r09) * k02
								"smull2  v25.4s, v6.8h, %10.h[2]      \n"

								// r1
								"smlal  v20.4s, v8.4h, %10.h[3]       \n" // (r10 - r17) * k03
								"smlal2  v21.4s, v8.8h, %10.h[3]      \n"
								"smlal  v22.4s, v9.4h, %11.h[0]       \n" // (r11 - r18) * k04
								"smlal2  v23.4s, v9.8h, %11.h[0]      \n"
								"smlal  v24.4s, v10.4h, %11.h[1]      \n" // (r12 - r19) * k05
								"smlal2  v25.4s, v10.8h, %11.h[1]     \n"

								// r2
								"smlal  v20.4s, v12.4h, %11.h[2]      \n" // (r20 - r27) * k06
								"smlal2  v21.4s, v12.8h, %11.h[2]     \n"
								"smlal  v22.4s, v13.4h, %11.h[3]      \n" // (r21 - r28) * k07
								"smlal2  v23.4s, v13.8h, %11.h[3]     \n"
								"smlal  v24.4s, v14.4h, %12.h[0]      \n" // (r22 - r29) * k08
								"smlal2  v25.4s, v14.8h, %12.h[0]     \n"

								// add and save
								"add    v20.4s, v20.4s, v22.4s        \n"
								"add    v21.4s, v21.4s, v23.4s        \n"
								"add    v20.4s, v20.4s, v24.4s        \n"
								"add    v21.4s, v21.4s, v25.4s        \n"

								"st1    {v20.4s, v21.4s}, [%1], #32   \n"

								"subs   %w0, %w0, #1                  \n"
								"bne    0b                            \n"

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
								"w"(_k0123), // %10
								"w"(_k4567), // %11
								"w"(_k8xxx)  // %12
								: "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25");
						}
#else
						if (nn > 0)
						{
							asm volatile(
								"0:                              \n"
								// r0
								"vld2.s8    {d30-d31}, [%2]!     \n" // r0
								"vld2.s8    {d10-d11}, [%2]      \n"
								"vext.s8    d12, d30, d10, #1    \n"

								"vmovl.s8    q5, d31             \n" // r01
								"vmovl.s8    q15, d30            \n" // r00
								"vmovl.s8    q6, d12             \n" // r02
								// sum0
								"vmull.s16  q7, d30, %P10[0]     \n" // (r00 - r07) * k00
								"vmull.s16  q8, d31, %P10[0]     \n"
								"vmull.s16  q9, d10, %P10[1]     \n" // (r01 - r08) * k01
								"vmull.s16  q10, d11, %P10[1]    \n"
								"vmlal.s16  q7, d12, %P10[2]     \n" // (r02 - r09) * k02
								"vmlal.s16  q8, d13, %P10[2]     \n"

								// r1
								"vld2.s8    {d30-d31}, [%3]!     \n" // r1
								"vld2.s8    {d10-d11}, [%3]      \n"
								"vext.s8    d12, d30, d10, #1    \n"

								"vmovl.s8    q5, d31             \n" // r11
								"vmovl.s8    q15, d30            \n" // r10
								"vmovl.s8    q6, d12             \n" // r12
								// sum0
								"vmlal.s16  q7, d30, %P10[3]     \n" // (r10 - r17) * k03
								"vmlal.s16  q8, d31, %P10[3]     \n"
								"vmlal.s16  q9, d10, %P11[0]     \n" // (r11 - r18) * k04
								"vmlal.s16  q10, d11, %P11[0]    \n"
								"vmlal.s16  q7, d12, %P11[1]     \n" // (r12 - r19) * k05
								"vmlal.s16  q8, d13, %P11[1]     \n"

								// r2
								"vld2.s8    {d30-d31}, [%4]!     \n" // r2
								"vld2.s8    {d10-d11}, [%4]      \n"
								"vext.s8    d12, d30, d10, #1    \n"

								"vmovl.s8    q5, d31             \n" // r21
								"vmovl.s8    q15, d30            \n" // r20
								"vmovl.s8    q6, d12             \n" // r22

								// sum0
								"vmlal.s16  q7, d30, %P11[2]     \n" // (r20 - r27) * k06
								"vmlal.s16  q8, d31, %P11[2]     \n"
								"vmlal.s16  q9, d10, %P11[3]     \n" // (r21 - r28) * k07
								"vmlal.s16  q10, d11, %P11[3]    \n"
								"vmlal.s16  q7, d12, %P12[0]     \n" // (r22 - r29) * k08
								"vmlal.s16  q8, d13, %P12[0]     \n"

								"subs   %0, %0, #1               \n"

								// add and save
								"vadd.s32    q7, q7, q9          \n"
								"vadd.s32    q8, q8, q10         \n"

								"vst1.s32    {d14-d17}, [%1]!    \n"

								"bne    0b                       \n"

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
								"w"(_k0123), // %10
								"w"(_k4567), // %11
								"w"(_k8xxx)  // %12
								: "cc", "memory", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
						}
#endif // __aarch64__
#endif // __ARM_NEON
						for (; remain > 0; remain--)
						{
							int sum = 0;

							sum += (int)r0[0] * kernel[0];
							sum += (int)r0[1] * kernel[1];
							sum += (int)r0[2] * kernel[2];
							sum += (int)r1[0] * kernel[3];
							sum += (int)r1[1] * kernel[4];
							sum += (int)r1[2] * kernel[5];
							sum += (int)r2[0] * kernel[6];
							sum += (int)r2[1] * kernel[7];
							sum += (int)r2[2] * kernel[8];

							*outptr = sum;

							r0 += 2;
							r1 += 2;
							r2 += 2;
							outptr++;
						}

						r0 += tailstep;
						r1 += tailstep;
						r2 += tailstep;
					}
				}
			}
		}

		INSTANCE_CLASS(operation_convolutiondepthwise_arm);
		REGISTE(operation_convolutiondepthwise_arm);
	}
}