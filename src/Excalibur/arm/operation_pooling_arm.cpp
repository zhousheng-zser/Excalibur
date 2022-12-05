#include "../../../include/Excalibur/arm/operation_pooling_arm.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"
#include "../../../include/Excalibur/operation_make_border.hpp"
#include <cfloat>

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_pooling_arm<Dtype>::operation_pooling_arm(const operation_param& param) : operation<Dtype>(param)
		{
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					type_ = (pooling_type)atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "1")
				{
					kernel_size_w_ = atoi(split_string(attrs[i], "=")[1].c_str());
					kernel_size_h_ = kernel_size_w_;
				}
				else if (split_string(attrs[i], "=")[0] == "2")
				{
					stride_w_ = atoi(split_string(attrs[i], "=")[1].c_str());
					stride_h_ = stride_w_;
				}
				else if (split_string(attrs[i], "=")[0] == "3")
				{
					pad_left_ = atoi(split_string(attrs[i], "=")[1].c_str());
					pad_right_ = pad_left_;
					pad_top_ = pad_left_;
					pad_bottom_ = pad_left_;
				}
				else if (split_string(attrs[i], "=")[0] == "4")
				{
					global_pooling_ = (bool)atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "5")
				{
					pad_mode_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "11")
				{
					kernel_size_h_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "12")
				{
					stride_h_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "13")
				{
					pad_top_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "14")
				{
					pad_right_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "15")
				{
					pad_bottom_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "-23330")
				{
					//do nothing
				}
				else
				{
					LOG(FATAL) << "Un-supported Pooling Attribution " << split_string(attrs[i], "=")[0];
				}
			}
		}

		template<typename Dtype>
		void operation_pooling_arm<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms, std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), tops.size());

			for (size_t i = 0; i < bottoms.size(); i++)
			{
				memory::orderType order = bottoms[i]->order();
				if (order == memory::NHWC)
					bottoms[i]->convert_order();

				if (((kernel_size_w_ != 2 && kernel_size_h_ != 2)&& (kernel_size_w_ != 3 && kernel_size_h_ != 3)) || (stride_w_ != 2 || stride_h_ != 2) || type_ != pooling_type::MAX)
				{
					NOT_IMPLEMENTED;
				}

				int num = bottoms[i]->num();
				int w = bottoms[i]->width();
				int h = bottoms[i]->height();
				int channels = bottoms[i]->channels();

				std::shared_ptr<memory::tensor<float> > bottom_bordered = bottoms[i];

				int wtailpad = 0, htailpad = 0;
				if (pad_mode_ == 0)
				{
					int wtail = (w + pad_left_ + pad_right_ - kernel_size_w_) % stride_w_;
					wtailpad = stride_w_ - wtail;
					int htail = (h + pad_top_ + pad_bottom_ - kernel_size_h_) % stride_h_;
					htailpad = stride_h_ - htail;
				}

				make_border(bottoms[i], bottom_bordered, pad_top_, pad_bottom_ + htailpad, pad_left_, pad_right_ + wtailpad, border_constant, -FLT_MAX);
				w = bottom_bordered->width();
				h = bottom_bordered->height();

				int outw = (w - kernel_size_w_) / stride_w_ + 1;
				int outh = (h - kernel_size_h_) / stride_h_ + 1;

				tops[i].reset(new memory::tensor<float>(std::vector<int>{num, channels, outh, outw}, bottoms[i]->device(), memory::NCHW, bottoms[i]->allocator()));

				if (kernel_size_w_ == 2 && kernel_size_h_ == 2)
					pooling2x2s2_max_neon(bottom_bordered, tops[i]);
				else if (kernel_size_w_ == 3 && kernel_size_h_ == 3)
					pooling3x3s2_max_neon(bottom_bordered, tops[i]);
			}
		}
		template<typename Dtype>
		void operation_pooling_arm<Dtype>::pooling2x2s2_max_neon(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top)
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

			for (int num_i = 0; num_i < num; num_i++)
			{
				const float *bottom_data = bottom->cpu_data() + num_i * inch * bottom_cstep;
				float *top_data = top->mutable_cpu_data() + num_i * outch * top_cstep;

#pragma omp parallel for num_threads(2) 
				for (int q = 0; q < inch; q++)
				{
					const float* img0 = bottom_data + (q)* bottom_cstep;
					float* outptr = top_data + (q)* top_cstep;

					const float* r0 = img0;
					const float* r1 = img0 + w;

					for (int i = 0; i < outh; i++)
					{
#if __ARM_NEON
						int nn = outw >> 2;
						int remain = outw - (nn << 2);
#else
						int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
						if (nn > 0)
						{
							asm volatile(
								"0:                                   \n"
								"prfm       pldl1keep, [%1, #256]     \n"
								"prfm       pldl1keep, [%2, #256]     \n"
								"ld1        {v0.4s, v1.4s}, [%1], #32 \n"
								"ld1        {v2.4s, v3.4s}, [%2], #32 \n"
								"fmax       v0.4s, v0.4s, v2.4s       \n"
								"fmax       v1.4s, v1.4s, v3.4s       \n"
								"fmaxp      v2.4s, v0.4s, v1.4s       \n"
								"subs       %w0, %w0, #1              \n"
								"st1        {v2.4s}, [%3], #16        \n"
								"bne        0b                        \n"
								: "=r"(nn),     // %0
								"=r"(r0),     // %1
								"=r"(r1),     // %2
								"=r"(outptr)  // %3
								: "0"(nn),
								"1"(r0),
								"2"(r1),
								"3"(outptr)
								: "cc", "memory", "v0", "v1", "v2", "v3"
								);
						}
#else
						if (nn > 0)
						{
							asm volatile(
								"0:                             \n"
								"pld        [%1, #256]          \n"
								"pld        [%2, #256]          \n"
								"vld1.f32   {d0-d3}, [%1]!      \n"
								"vld1.f32   {d4-d7}, [%2]!      \n"
								"vmax.f32   q0, q0, q2          \n"
								"vmax.f32   q1, q1, q3          \n"
								"vpmax.f32  d4, d0, d1          \n"
								"vpmax.f32  d5, d2, d3          \n"
								"subs       %0, #1              \n"
								"vst1.f32   {d4-d5}, [%3]!      \n"
								"bne        0b                  \n"
								: "=r"(nn),     // %0
								"=r"(r0),     // %1
								"=r"(r1),     // %2
								"=r"(outptr)  // %3
								: "0"(nn),
								"1"(r0),
								"2"(r1),
								"3"(outptr)
								: "cc", "memory", "q0", "q1", "q2", "q3"
								);
						}
#endif // __aarch64__
#endif // __ARM_NEON
						for (; remain > 0; remain--)
						{
							float max0 = std::max(r0[0], r0[1]);
							float max1 = std::max(r1[0], r1[1]);

							*outptr = std::max(max0, max1);

							r0 += 2;
							r1 += 2;
							outptr++;
						}

						r0 += tailstep;
						r1 += tailstep;
					}
				}
			}
		}
		template<typename Dtype>
		void operation_pooling_arm<Dtype>::pooling3x3s2_max_neon(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top)
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

			for (int num_i = 0; num_i < num; num_i++)
			{
				const float *bottom_data = bottom->cpu_data() + num_i * inch * bottom_cstep;
				float *top_data = top->mutable_cpu_data() + num_i * outch * top_cstep;

#pragma omp parallel for num_threads(2) 
				for (int q = 0; q < inch; q++)
				{
					const float* img0 = bottom_data + (q)* bottom_cstep;
					float* outptr = top_data + (q)* top_cstep;

					const float* r0 = img0;
					const float* r1 = img0 + w;
					const float* r2 = img0 + w * 2;

					for (int i = 0; i < outh; i++)
					{
#if __ARM_NEON
						int nn = outw >> 2;
						int remain = outw - (nn << 2);
#else
						int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
						if (nn > 0)
						{
							asm volatile(
								"prfm       pldl1keep, [%1, #256]       \n"
								"ld2        {v0.4s, v1.4s}, [%1], #32   \n"
								"prfm       pldl1keep, [%2, #256]       \n"
								"ld2        {v2.4s, v3.4s}, [%2], #32   \n"
								"prfm       pldl1keep, [%3, #256]       \n"
								"ld2        {v4.4s, v5.4s}, [%3], #32   \n"
								"0:                                     \n"

								"prfm       pldl1keep, [%1, #256]       \n"
								"ld2        {v6.4s, v7.4s}, [%1], #32   \n"

								"fmax       v12.4s, v0.4s, v1.4s        \n"
								"fmax       v13.4s, v2.4s, v3.4s        \n"

								"prfm       pldl1keep, [%2, #256]       \n"
								"ld2        {v8.4s, v9.4s}, [%2], #32   \n"

								"fmax       v14.4s, v4.4s, v5.4s        \n"
								"ext        v0.16b, v0.16b, v6.16b, #4  \n"

								"prfm       pldl1keep, [%3, #256]       \n"
								"ld2        {v10.4s, v11.4s}, [%3], #32 \n"

								"ext        v2.16b,  v2.16b, v8.16b, #4 \n"

								"fmax       v12.4s, v12.4s, v0.4s       \n"
								"ext        v4.16b, v4.16b, v10.16b, #4 \n"

								"fmax       v13.4s, v13.4s, v2.4s       \n"
								"fmax       v14.4s, v14.4s, v4.4s       \n"
								"fmax       v12.4s, v12.4s, v13.4s      \n"

								"orr        v0.16b, v6.16b, v6.16b      \n"
								"orr        v1.16b, v7.16b, v7.16b      \n"
								"fmax       v12.4s, v12.4s, v14.4s      \n"

								"orr        v2.16b, v8.16b, v8.16b      \n"
								"orr        v3.16b, v9.16b, v9.16b      \n"
								"orr        v4.16b, v10.16b, v10.16b    \n"
								"orr        v5.16b, v11.16b, v11.16b    \n"

								"subs       %w0, %w0, #1                \n"
								"st1        {v12.4s}, [%4], #16         \n"
								"bne        0b                          \n"
								"sub        %1, %1, #32                 \n"
								"sub        %2, %2, #32                 \n"
								"sub        %3, %3, #32                 \n"
								: "=r"(nn),     // %0
								"=r"(r0),     // %1
								"=r"(r1),     // %2
								"=r"(r2),     // %3
								"=r"(outptr)  // %4
								: "0"(nn),
								"1"(r0),
								"2"(r1),
								"3"(r2),
								"4"(outptr)
								: "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14"
								);
						}
#else
						if (nn > 0)
						{
							asm volatile(
								"pld        [%1, #256]          \n"
								"vld2.f32   {d0-d3}, [%1]!      \n"// q0 = 0 2 4 6  q1 = 1 3 5 7
								"pld        [%2, #256]          \n"
								"vld2.f32   {d4-d7}, [%2]!      \n"
								"pld        [%3, #256]          \n"
								"vld2.f32   {d8-d11}, [%3]!     \n"
								"0:                             \n"
								"pld        [%1, #256]          \n"
								"vld2.f32   {d12-d15}, [%1]!    \n"// q6 = 8 10 12 14  q7 = 9 11 13 15

								"vmax.f32   q12, q0, q1         \n"
								"vmax.f32   q13, q2, q3         \n"

								"pld        [%2, #256]          \n"
								"vld2.f32   {d16-d19}, [%2]!    \n"

								"vmax.f32   q14, q4, q5         \n"
								"vext.32    q0, q0, q6, #1      \n"

								"pld        [%3, #256]          \n"
								"vld2.f32   {d20-d23}, [%3]!    \n"

								"vext.32    q2, q2, q8, #1      \n"

								"vmax.f32   q12, q12, q0        \n"
								"vext.32    q4, q4, q10, #1     \n"

								"vmax.f32   q13, q13, q2        \n"
								"vmax.f32   q14, q14, q4        \n"
								"vmax.f32   q12, q12, q13       \n"

								"vorr       q0, q6, q6          \n"
								"vorr       q1, q7, q7          \n"
								"vmax.f32   q12, q12, q14       \n"

								"vorr       q2, q8, q8          \n"
								"vorr       q3, q9, q9          \n"
								"vorr       q4, q10, q10        \n"
								"vorr       q5, q11, q11        \n"

								"subs       %0, #1              \n"
								"vst1.f32   {d24-d25}, [%4]!    \n"
								"bne        0b                  \n"
								"sub        %1, #32             \n"
								"sub        %2, #32             \n"
								"sub        %3, #32             \n"
								: "=r"(nn),     // %0
								"=r"(r0),     // %1
								"=r"(r1),     // %2
								"=r"(r2),     // %3
								"=r"(outptr)  // %4
								: "0"(nn),
								"1"(r0),
								"2"(r1),
								"3"(r2),
								"4"(outptr)
								: "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14"
								);
						}
#endif // __aarch64__
#endif // __ARM_NEON
						for (; remain > 0; remain--)
						{
							float max0 = std::max(std::max(r0[0], r0[1]), r0[2]);
							float max1 = std::max(std::max(r1[0], r1[1]), r1[2]);
							float max2 = std::max(std::max(r2[0], r2[1]), r2[2]);

							*outptr = std::max(std::max(max0, max1), max2);

							r0 += 2;
							r1 += 2;
							r2 += 2;
							outptr++;
						}

						r0 += tailstep;//1 + w;
						r1 += tailstep;//1 + w;
						r2 += tailstep;//1 + w;
					}
				}
			}
		}

		INSTANCE_CLASS(operation_pooling_arm);
//		REGISTE(operation_pooling_arm);
	}
}