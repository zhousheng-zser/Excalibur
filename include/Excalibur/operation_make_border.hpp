#pragma once
#ifndef _OPERATION_MAKE_BORDER_HPP_
#define _OPERATION_MAKE_BORDER_HPP_
#include <memory>
#include <Primitives/tensor.hpp>
namespace glasssix
{
	namespace excalibur
	{
		enum border_type { border_constant, border_replicate };

		/// <summary>
		/// expand border
		/// </summary>
		/// <param name="src">original memory::tensor</param>
		/// <param name="dst">new memory::tensor</param>
		/// <param name="top">pixels to expand at top of image</param>
		/// <param name="bottom">pixels to expand at bottom of image</param>
		/// <param name="left">pixels to expand at left of image</param>
		/// <param name="right">pixels to expand at right of image</param>
		/// <param name="type">borderType: Border_Constant(default, use constant pixel value(fill_pixel_value) to fill in new blank area) / Border_Replicate(replicate neighboring pixel to fill in new blank area)</param>
		/// <param name="fill_pixel_value">validate when borderType is Border_Constant, zero by default</param>
		template <typename Dtype>
		static void make_border(const std::shared_ptr<memory::tensor<Dtype>> &src, std::shared_ptr<memory::tensor<Dtype>>& dst,
			int top, int bottom, int left, int right, border_type type = border_constant, Dtype fill_pixel_value = 0)
		{
			/*if (src->device() >= 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_gpu() instead!!!";
				return;
			}*/

			int num = src->num();
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			int src_offset = height * width;
			int src_num_offset = channels * height * width;

			int dst_height = height + top + bottom;
			int dst_width = width + left + right;
			int dst_offset = dst_height * dst_width;
			int dst_num_offset = channels * dst_height * dst_width;

			if (top < 0 || bottom < 0 || left < 0 || right < 0)
			{
				LOG(ERROR) << "top, bottom, left, right: should all be non-negtive.";
				return;
			}

			if (dst_height == height && dst_width == width)
			{
				dst = std::make_shared<memory::tensor<Dtype>>(src->clone());
				return;
			}

			std::shared_ptr<memory::tensor<Dtype>> dst_temp;
			if (src->order() == memory::NCHW)
			{
				dst_temp.reset(new memory::tensor<Dtype>(std::vector<int>{num, channels, dst_height, dst_width}, src->device(), src->order(), src->allocator()));
				Dtype* dst_data = dst_temp->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

				if (type == border_constant)
				{
					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
						for (int ch = 0; ch < channels; ++ch)
						{
							int src_channel_offset = ch * src_offset;
							int dst_channel_offset = ch * dst_offset;
							
							//top
							for (int row = 0; row < top; row++)
							{
								int dst_index = dst_channel_offset + row * dst_width;
								for (int col = 0; col < dst_width; col++)
								{
									dst_data[dst_n_offset + dst_index + col] = fill_pixel_value;
								}
							}

							//center
							for (int row = top; row < top + height; ++row)
							{
								int src_index = src_channel_offset + (row - top) * width;
								int dst_index = dst_channel_offset + row * dst_width;

								for (int col = 0; col < left; col++)
								{
									dst_data[dst_n_offset + dst_index + col] = fill_pixel_value;
								}

								memcpy(dst_data + dst_n_offset + dst_index + left, src_data + src_n_offset + src_index, width * sizeof(Dtype));

								for (int col = left + width; col < dst_width; col++)
								{
									dst_data[dst_n_offset + dst_index + col] = fill_pixel_value;
								}
							}

							//bottom
							for (int row = top + height; row < dst_height; row++)
							{
								int dst_index = dst_channel_offset + row * dst_width;
								for (int col = 0; col < dst_width; col++)
								{
									dst_data[dst_n_offset + dst_index + col] = fill_pixel_value;
								}
							}
							
						}
					}
				}
				else if (type == border_replicate)
				{
					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;
						for (int ch = 0; ch < channels; ++ch)
						{
							int src_channel_offset = ch * src_offset;
							int dst_channel_offset = ch * dst_offset;

							//top
							for (int row = 0; row < top; ++row)
							{
								int dst_index = dst_channel_offset + row * dst_width;

								//left
								for (int col = 0; col < left; col++)
								{
									dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_channel_offset];
								}

								//center
								memcpy(dst_data + dst_n_offset + dst_index + left, src_data + src_n_offset + src_channel_offset, width * sizeof(Dtype));

								//right
								for (int col = left + width; col < dst_width; col++)
								{
									dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_channel_offset + width - 1];
								}
							}

							//center
							for (int row = top; row < top + height; ++row)
							{
								int src_index = src_channel_offset + (row - top) * width;
								int dst_index = dst_channel_offset + row * dst_width;

								//left
								for (int col = 0; col < left; col++)
								{
									dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_index];
								}

								//center
								memcpy(dst_data + dst_n_offset + dst_index + left, src_data + src_n_offset + src_index, width * sizeof(Dtype));

								//right
								for (int col = left + width; col < dst_width; col++)
								{
									dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_index + width - 1];
								}
							}

							//bottom
							for (int row = top + height; row < dst_height; ++row)
							{
								int src_index = src_channel_offset + (height - 1) * width;
								int dst_index = dst_channel_offset + row * dst_width;

								//left
								for (int col = 0; col < left; col++)
								{
									dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_index];
								}

								//center
								memcpy(dst_data + dst_n_offset + dst_index + left, src_data + src_n_offset + src_index, width * sizeof(Dtype));

								//right
								for (int col = left + width; col < dst_width; col++)
								{
									dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_index + width - 1];
								}
							}
						}
					}
				}
				else
				{
					LOG(ERROR) << "Un-support border type.";
					return;
				}
			}
			else if (src->order() == memory::NHWC)
			{
				dst_temp.reset(new memory::tensor<Dtype>(std::vector<int>{num, dst_height, dst_width, channels}, src->device(), src->order()));
				Dtype* dst_data = dst_temp->mutable_cpu_data();
				const Dtype* src_data = src->cpu_data();

				if (type == border_constant)
				{
					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;

						//top
						for (int row = 0; row < top; row++)
						{
							int dst_index1 = row * dst_width * channels;
							for (int col = 0; col < dst_width; col++)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ch++)
								{
									dst_data[dst_n_offset + dst_index2 + ch] = fill_pixel_value;
								}
							}
						}

						//center
						for (int row = top; row < top + height; ++row)
						{
							int src_index = (row - top) * width * channels;

							//left
							int dst_index1 = row * dst_width * channels;
							for (int col = 0; col < left; col++)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ch++)
								{
									dst_data[dst_n_offset + dst_index2 + ch] = fill_pixel_value;
								}
							}

							//center
							memcpy(dst_data + dst_n_offset + dst_index1 + left * channels, src_data + src_n_offset + src_index, width * channels * sizeof(Dtype));

							//right
							for (int col = left + width; col < dst_width; col++)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ch++)
								{
									dst_data[dst_n_offset + dst_index2 + ch] = fill_pixel_value;
								}
							}
						}

						//bottom
						for (int row = top + height; row < dst_height; row++)
						{
							int dst_index1 = row * dst_width * channels;
							for (int col = 0; col < dst_width; col++)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ch++)
								{
									dst_data[dst_n_offset + dst_index2 + ch] = fill_pixel_value;
								}
							}
						}
					}
				}
				else if (type == border_replicate)
				{
					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;

						//top
						for (int row = 0; row < top; ++row)
						{
							int dst_index1 = row * dst_width * channels;

							//left
							for (int col = 0; col < left; ++col)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + ch];
								}
							}

							//center
							memcpy(dst_data + dst_n_offset + dst_index1 + left * channels, src_data + src_n_offset, width * channels * sizeof(Dtype));

							//right
							for (int col = left + width; col < dst_width; ++col)
							{
								int dst_index2 = dst_index1 + col * channels;
								int src_index = (width - 1) * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + src_index + ch];
								}
							}
						}


						//center
						for (int row = top; row < top + height; ++row)
						{
							int src_index1 = (row - top) * width * channels;
							int dst_index1 = row * dst_width * channels;

							//left
							for (int col = 0; col < left; ++col)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + src_index1 + ch];
								}
							}

							//center
							memcpy(dst_data + dst_n_offset + dst_index1 + left * channels, src_data + src_n_offset + src_index1, width * channels * sizeof(Dtype));

							//right
							int src_index2 = src_index1 + (width - 1) * channels;
							for (int col = left + width; col < dst_width; ++col)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + src_index2 + ch];
								}
							}
						}


						//bottom
						for (int row = top + height; row < dst_height; ++row)
						{
							int dst_index1 = row * dst_width * channels;
							int src_index1 = (height - 1) * width * channels;

							//left
							for (int col = 0; col < left; ++col)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + src_index1 + ch];
								}
							}

							//center
							memcpy(dst_data + dst_n_offset + dst_index1 + left * channels, src_data + src_n_offset + src_index1, width * channels * sizeof(Dtype));

							//right
							int src_index2 = src_index1 + (width - 1) * channels;
							for (int col = left + width; col < dst_width; ++col)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + src_index2 + ch];
								}
							}
						}
					}
				}
				else
				{
					LOG(ERROR) << "Un-support border type.";
					return;
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			dst = std::make_shared<memory::tensor<Dtype>>(dst_temp->clone());
		}


#ifdef __ARM_NEON
		static void padding_constant_pack4_neon(const float* src, int src_h, int src_w, float* dst, int dst_h, int dst_w, int top, int bottom, int left, int right, float32x4_t v)
		{
			const float* ptr = src;
			float* outptr = dst;

			int w = src_w;
			int h = src_h;

			int top_size = top * dst_w;
			int bottom_size = bottom * dst_w;

#if __aarch64__
			asm volatile(
				"mov    v0.16b, %10.16b         \n"
				"mov    v1.16b, %10.16b         \n"
				"mov    v2.16b, %10.16b         \n"
				"mov    v3.16b, %10.16b         \n"

				// fill top
				"lsr    w4, %w8, #3             \n" // w4 = nn = top_size >> 3
				"cmp    w4, #0                  \n"
				"beq    1f                      \n"

				"0:                             \n"
				"st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
				"subs   w4, w4, #1              \n"
				"st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
				"bne    0b                      \n"

				"1:                             \n"

				// fill top remain
				"and    w4, %w8, #7             \n" // w4 = remain = top_size & 7

				"cmp    w4, #4                  \n" // w4 >= 4
				"blt    2f                      \n"
				"sub    w4, w4, #4              \n"
				"st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
				"2:                             \n"

				"cmp    w4, #2                  \n" // w4 >= 2
				"blt    3f                      \n"
				"sub    w4, w4, #2              \n"
				"st1    {v0.4s, v1.4s}, [%0], #32 \n"
				"3:                             \n"

				"cmp    w4, #0                  \n" // w4 > 0
				"beq    4f                      \n"
				"st1    {v0.4s}, [%0], #16      \n"
				"4:                             \n"

				// fill center h loop
				"cmp    %w5, #0                 \n"
				"beq    15f                     \n"
				"5:                             \n"

				// fill left
				"mov    w4, %w6                 \n" // w4 = left
				"cmp    w4, #0                  \n"
				"beq    7f                      \n"

				"6:                             \n"
				"st1    {v0.4s}, [%0], #16      \n"
				"subs   w4, w4, #1              \n"
				"bne    6b                      \n"

				"7:                             \n"

				// fill middle
				"lsr    w4, %w4, #3             \n" // w4 = nn = w >> 3
				"cmp    w4, #0                  \n"
				"beq    9f                      \n"

				"8:                             \n"
				"prfm   pldl1keep, [%1, #512]   \n"
				"ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
				"prfm   pldl1keep, [%1, #512]   \n"
				"ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%1], #64 \n"
				"subs   w4, w4, #1              \n"
				"st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
				"st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
				"bne    8b                      \n"

				"9:                             \n"

				"and    w4, %w4, #7             \n" // w4 = remain = w & 7

				"cmp    w4, #4                  \n" // w4 >= 4
				"blt    10f                     \n"
				"prfm   pldl1keep, [%1, #512]   \n"
				"ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
				"sub    w4, w4, #4              \n"
				"st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
				"10:                            \n"

				"cmp    w4, #2                  \n" // w4 >= 2
				"blt    11f                     \n"
				"prfm   pldl1keep, [%1, #256]   \n"
				"ld1    {v16.4s, v17.4s}, [%1], #32 \n"
				"sub    w4, w4, #2              \n"
				"st1    {v16.4s, v17.4s}, [%0], #32 \n"
				"11:                            \n"

				"cmp    w4, #0                  \n" // w4 > 0
				"beq    12f                     \n"
				"prfm   pldl1keep, [%1, #128]   \n"
				"ld1    {v16.4s}, [%1], #16     \n"
				"st1    {v16.4s}, [%0], #16     \n"
				"12:                            \n"

				// fill right
				"mov    w4, %w7                 \n" // w4 = right
				"cmp    w4, #0                  \n"
				"beq    14f                     \n"

				"13:                            \n"
				"subs   w4, w4, #1              \n"
				"st1    {v0.4s}, [%0], #16      \n"
				"bne    13b                     \n"
				"14:                            \n"

				"subs   %w5, %w5, #1            \n"
				"bne    5b                      \n"

				"15:                            \n"

				// fill bottom
				"lsr    w4, %w9, #3             \n" // w4 = nn = bottom_size >> 3
				"cmp    w4, #0                  \n"
				"beq    17f                     \n"

				"16:                            \n"
				"st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
				"subs   w4, w4, #1              \n"
				"st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
				"bne    16b                     \n"
				"17:                            \n"

				// fill bottom remain
				"and    w4, %w9, #7             \n" // w4 = remain = bottom_size & 7

				"cmp    w4, #4                  \n" // w4 >= 4
				"blt    18f                     \n"
				"sub    w4, w4, #4              \n"
				"st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
				"18:                            \n"

				"cmp    w4, #2                  \n" // w4 >= 2
				"blt    19f                     \n"
				"sub    w4, w4, #2              \n"
				"st1    {v0.4s, v1.4s}, [%0], #32 \n"
				"19:                            \n"

				"cmp    w4, #0                  \n" // w4 > 0
				"beq    20f                     \n"
				"st1    {v0.4s}, [%0], #16      \n"
				"20:                            \n"

				: "=r"(outptr), // %0
				"=r"(ptr)     // %1
				: "0"(outptr),
				"1"(ptr),
				"r"(w),           // %4
				"r"(h),           // %5
				"r"(left),        // %6
				"r"(right),       // %7
				"r"(top_size),    // %8
				"r"(bottom_size), // %9
				"w"(v)            // %10
				: "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else  // __aarch64__
			asm volatile(
				"vmov       q0, %q10            \n"
				"vmov       q1, %q10            \n"
				"vmov       q2, %q10            \n"
				"vmov       q3, %q10            \n"

				// fill top
				"lsr        r4, %8, #3          \n" // r4 = nn = top_size >> 3
				"cmp        r4, #0              \n"
				"beq        1f                  \n"

				"0:                             \n"
				"vstm       %0!, {d0-d7}        \n"
				"subs       r4, r4, #1          \n"
				"vstm       %0!, {d0-d7}        \n"
				"bne        0b                  \n"

				"1:                             \n"

				// fill top remain
				"and        r4, %8, #7          \n" // r4 = remain = top_size & 7

				"cmp        r4, #4              \n" // r4 >= 4
				"blt        2f                  \n"
				"sub        r4, r4, #4          \n"
				"vstm       %0!, {d0-d7}        \n"
				"2:                             \n"

				"cmp        r4, #2              \n" // r4 >= 2
				"blt        3f                  \n"
				"sub        r4, r4, #2          \n"
				"vst1.f32   {d0-d3}, [%0 :128]! \n"
				"3:                             \n"

				"cmp        r4, #0              \n" // r4 > 0
				"beq        4f                  \n"
				"vst1.f32   {d0-d1}, [%0 :128]! \n"
				"4:                             \n"

				// fill center h loop
				"cmp        %5, #0              \n"
				"beq        15f                 \n"
				"5:                             \n"

				// fill left
				"mov        r4, %6              \n" // r4 = left
				"cmp        r4, #0              \n"
				"beq        7f                  \n"

				"6:                             \n"
				"vst1.f32   {d0-d1}, [%0 :128]! \n"
				"subs       r4, r4, #1          \n"
				"bne        6b                  \n"

				"7:                             \n"

				// fill middle
				"lsr        r4, %4, #3          \n" // r4 = nn = w >> 3
				"cmp        r4, #0              \n"
				"beq        9f                  \n"

				"8:                             \n"
				"pld        [%1, #512]          \n"
				"vldm       %1!, {d16-d23}      \n"
				"pld        [%1, #512]          \n"
				"vldm       %1!, {d24-d31}      \n"
				"subs       r4, r4, #1          \n"
				"vstm       %0!, {d16-d23}      \n"
				"vstm       %0!, {d24-d31}      \n"
				"bne        8b                  \n"

				"9:                             \n"

				"and        r4, %4, #7          \n" // r4 = remain = w & 7

				"cmp        r4, #4              \n" // r4 >= 4
				"blt        10f                 \n"
				"pld        [%1, #512]          \n"
				"vldm       %1!, {d16-d23}      \n"
				"sub        r4, r4, #4          \n"
				"vstm       %0!, {d16-d23}      \n"
				"10:                            \n"

				"cmp        r4, #2              \n" // r4 >= 2
				"blt        11f                 \n"
				"pld        [%1, #256]          \n"
				"vld1.f32   {d16-d19}, [%1 :128]! \n"
				"sub        r4, r4, #2          \n"
				"vst1.f32   {d16-d19}, [%0 :128]! \n"
				"11:                            \n"

				"cmp        r4, #0              \n" // r4 > 0
				"beq        12f                 \n"
				"pld        [%1, #128]          \n"
				"vld1.f32   {d16-d17}, [%1 :128]! \n"
				"vst1.f32   {d16-d17}, [%0 :128]! \n"
				"12:                            \n"

				// fill right
				"mov        r4, %7              \n" // r4 = right
				"cmp        r4, #0              \n"
				"beq        14f                 \n"

				"13:                            \n"
				"subs       r4, r4, #1          \n"
				"vst1.f32   {d0-d1}, [%0 :128]! \n"
				"bne        13b                 \n"
				"14:                            \n"

				"subs       %5, %5, #1          \n"
				"bne        5b                  \n"

				"15:                            \n"

				// fill bottom
				"lsr        r4, %9, #3          \n" // r4 = nn = bottom_size >> 3
				"cmp        r4, #0              \n"
				"beq        17f                 \n"

				"16:                            \n"
				"vstm       %0!, {d0-d7}        \n"
				"subs       r4, r4, #1          \n"
				"vstm       %0!, {d0-d7}        \n"
				"bne        16b                 \n"
				"17:                            \n"

				// fill bottom remain
				"and        r4, %9, #7          \n" // r4 = remain = bottom_size & 7

				"cmp        r4, #4              \n" // r4 >= 4
				"blt        18f                 \n"
				"sub        r4, r4, #4          \n"
				"vstm       %0!, {d0-d7}        \n"
				"18:                            \n"

				"cmp        r4, #2              \n" // r4 >= 2
				"blt        19f                 \n"
				"sub        r4, r4, #2          \n"
				"vst1.f32   {d0-d3}, [%0 :128]! \n"
				"19:                            \n"

				"cmp        r4, #0              \n" // r4 > 0
				"beq        20f                 \n"
				"vst1.f32   {d0-d1}, [%0 :128]! \n"
				"20:                            \n"

				: "=r"(outptr), // %0
				"=r"(ptr)     // %1
				: "0"(outptr),
				"1"(ptr),
				"r"(w),           // %4
				"r"(h),           // %5
				"r"(left),        // %6
				"r"(right),       // %7
				"r"(top_size),    // %8
				"r"(bottom_size), // %9
				"w"(v)            // %10
				: "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
		}

		static void make_border_pack4(const std::shared_ptr<memory::tensor<float>>& src, std::shared_ptr<memory::tensor<float>>& dst,
			int top, int bottom, int left, int right, border_type type = border_constant, float fill_pixel_value = 0)
		{
			if (top == 0 && bottom == 0 && left == 0 && right == 0)
			{
				dst = src;
				return;
			}

			int num = src->num();
			int w = src->width() / 4;
			int h = src->height();
			int channels = src->channels();

			int outw = w + left + right;
			int outh = h + top + bottom;
			int outc = channels;
			dst.reset(new memory::tensor<float>(std::vector<int>{num, outc, outh, outw * 4}, src->device(), src->order(), src->allocator()));

			float32x4_t pad_value = vdupq_n_f32(fill_pixel_value);
			for (size_t n = 0; n < num; n++)
			{
				const float* src_data = src->cpu_data() + n * src->count(1, 4);
				float* dst_data = dst->mutable_cpu_data() + n * dst->count(1, 4);

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
				for (int q = 0; q < outc; q++)
				{
					const float* src_data_q = src_data + h * w * 4;
					float* borderm = dst_data + outh * outw * 4;
					if (type == border_constant)
						padding_constant_pack4_neon(src_data_q, h, w, borderm, outh, outw, top, bottom, left, right, pad_value);
					else
						NOT_IMPLEMENTED;
				}
			}
			}
#endif

#ifdef USE_CUDA
		/// <summary>
		/// expand border
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		/// <param name="top">pixels to expand at top of image</param>
		/// <param name="bottom">pixels to expand at bottom of image</param>
		/// <param name="left">pixels to expand at left of image</param>
		/// <param name="right">pixels to expand at right of image</param>
		/// <param name="type">borderType: Border_Constant(default, use constant pixel value(fill_pixel_value) to fill in new blank area) / Border_Replicate(replicate neighboring pixel to fill in new blank area)</param>
		/// <param name="fill_pixel_value">validate when borderType is Border_Constant, zero by default</param>
		template <typename Dtype>
		void make_border_gpu(const std::shared_ptr<memory::tensor<Dtype>>& src, std::shared_ptr<memory::tensor<Dtype>>& dst,
			int top, int bottom, int left, int right, border_type type = border_constant, Dtype fill_pixel_value = 0)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			int num = src->num();
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			int src_offset = height * width;
			int src_num_offset = channels * height * width;

			int dst_height = height + top + bottom;
			int dst_width = width + left + right;
			int dst_offset = dst_height * dst_width;
			int dst_num_offset = channels * dst_height * dst_width;

			if (top < 0 || bottom < 0 || left < 0 || right < 0)
			{
				LOG(ERROR) << "top, bottom, left, right: should all be non-negtive.";
				return;
			}

			if (dst_height == height && dst_width == width)
			{
				dst = std::make_shared<memory::tensor<Dtype>>(src->clone());
				return;
			}

			std::shared_ptr<memory::tensor<Dtype>> dst_temp;
			if (src->order() == memory::NCHW)
			{
				dst_temp.reset(new memory::tensor<Dtype>(std::vector<int>{num, channels, dst_height, dst_width}, src->device(), src->order(), src->allocator()));
				Dtype* dst_data = dst_temp->mutable_gpu_data();
				const Dtype* src_data = src->gpu_data();

				if (type == border_constant)
				{
					std::vector<Dtype> top_data(top * dst_width, fill_pixel_value);
					std::vector<Dtype> center_left_data(left, fill_pixel_value);
					std::vector<Dtype> center_right_data(right, fill_pixel_value);
					std::vector<Dtype> bottom_data(bottom * dst_width, fill_pixel_value);

					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;

						for (int ch = 0; ch < channels; ++ch)
						{
							int src_channel_offset = ch * src_offset;
							int dst_channel_offset = ch * dst_offset;

							//top
							CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_channel_offset, top_data.data(), top * dst_width * sizeof(Dtype), cudaMemcpyDefault));

							//center
							for (int row = top; row < top + height; ++row)
							{
								int src_index = src_channel_offset + (row - top) * width;
								int dst_index = dst_channel_offset + row * dst_width;

								CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_index, center_left_data.data(), left * sizeof(Dtype), cudaMemcpyDefault));

								CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_index + left, src_data + src_n_offset + src_index, width * sizeof(Dtype), cudaMemcpyDefault));

								CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_index + left + width, center_right_data.data(), right * sizeof(Dtype), cudaMemcpyDefault));
							}

							//bottom
							CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_channel_offset + (top + height) * dst_width, bottom_data.data(), bottom * dst_width * sizeof(Dtype), cudaMemcpyDefault));
						}
					}
				}
				else if (type == border_replicate)
				{
					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;

						for (int ch = 0; ch < channels; ++ch)
						{
							int src_channel_offset = ch * src_offset;
							int dst_channel_offset = ch * dst_offset;

							//top
							for (int row = 0; row < top; ++row)
							{
								int dst_index = dst_channel_offset + row * dst_width;

								//left
								for (int col = 0; col < left; col++)
								{
									dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_channel_offset];
								}

								//center
								CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_index + left, src_data + src_n_offset + src_channel_offset, width * sizeof(Dtype), cudaMemcpyDefault));

								//right
								for (int col = left + width; col < dst_width; col++)
								{
									dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_channel_offset + width - 1];
								}
							}

							//center
							for (int row = top; row < top + height; ++row)
							{
								int src_index = src_channel_offset + (row - top) * width;
								int dst_index = dst_channel_offset + row * dst_width;

								//left
								for (int col = 0; col < left; col++)
								{
									dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_index];
								}

								//center
								CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_index + left, src_data + src_n_offset + src_index, width * sizeof(Dtype), cudaMemcpyDefault));

								//right
								for (int col = left + width; col < dst_width; col++)
								{
									dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_index + width - 1];
								}
							}

							//bottom
							for (int row = top + height; row < dst_height; ++row)
							{
								int src_index = src_channel_offset + (height - 1) * width;
								int dst_index = dst_channel_offset + row * dst_width;

								//left
								for (int col = 0; col < left; col++)
								{
									dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_index];
								}

								//center
								CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_index + left, src_data + src_n_offset + src_index, width * sizeof(Dtype), cudaMemcpyDefault));

								//right
								for (int col = left + width; col < dst_width; col++)
								{
									dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_index + width - 1];
								}
							}
						}
					}
				}
				else
				{
					LOG(ERROR) << "Un-support border type.";
					return;
				}
			}
			else if (src->order() == memory::NHWC)
			{
				dst_temp.reset(new memory::tensor<Dtype>(std::vector<int>{num, dst_height, dst_width, channels}, src->device(), src->order(), src->allocator()));
				Dtype* dst_data = dst_temp->mutable_gpu_data();
				const Dtype* src_data = src->gpu_data();

				if (type == border_constant)
				{
					std::vector<Dtype> top_data(channels * top * dst_width, fill_pixel_value);
					std::vector<Dtype> center_left_data(channels * left, fill_pixel_value);
					std::vector<Dtype> center_right_data(channels * right, fill_pixel_value);
					std::vector<Dtype> bottom_data(channels * bottom * dst_width, fill_pixel_value);

					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;

						//top
						CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset, top_data.data(), channels * top * dst_width * sizeof(Dtype), cudaMemcpyDefault));

						//center
						for (int row = top; row < top + height; ++row)
						{
							int src_index = (row - top) * width * channels;

							//left
							CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + row * dst_width * channels, center_left_data.data(), channels * left * sizeof(Dtype), cudaMemcpyDefault));

							//center
							CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + (row * dst_width + left) * channels, src_data + src_n_offset + src_index, width * channels * sizeof(Dtype), cudaMemcpyDefault));

							//right
							CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + (row * dst_width + left + width) * channels, center_right_data.data(), channels * right * sizeof(Dtype), cudaMemcpyDefault));
						}

						//bottom
						CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + (top + height) * dst_width * channels, bottom_data.data(), channels * bottom * dst_width * sizeof(Dtype), cudaMemcpyDefault));
					}
				}
				else if (type == border_replicate)
				{
					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;

						//top
						for (int row = 0; row < top; ++row)
						{
							int dst_index1 = row * dst_width * channels;

							//left
							for (int col = 0; col < left; ++col)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + ch];
								}
							}

							//center
							CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_index1 + left * channels, src_data + src_n_offset, width * channels * sizeof(Dtype), cudaMemcpyDefault));

							//right
							for (int col = left + width; col < dst_width; ++col)
							{
								int dst_index2 = dst_index1 + col * channels;
								int src_index = (width - 1) * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + src_index + ch];
								}
							}
						}


						//center
						for (int row = top; row < top + height; ++row)
						{
							int src_index1 = (row - top) * width * channels;
							int dst_index1 = row * dst_width * channels;

							//left
							for (int col = 0; col < left; ++col)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + src_index1 + ch];
								}
							}

							//center
							CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_index1 + left * channels, src_data + src_n_offset + src_index1, width * channels * sizeof(Dtype), cudaMemcpyDefault));

							//right
							int src_index2 = src_index1 + (width - 1) * channels;
							for (int col = left + width; col < dst_width; ++col)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + src_index2 + ch];
								}
							}
						}


						//bottom
						for (int row = top + height; row < dst_height; ++row)
						{
							int dst_index1 = row * dst_width * channels;
							int src_index1 = (height - 1) * width * channels;

							//left
							for (int col = 0; col < left; ++col)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + src_index1 + ch];
								}
							}

							//center
							CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_index1 + left * channels, src_data + src_n_offset + src_index1, width * channels * sizeof(Dtype), cudaMemcpyDefault));

							//right
							int src_index2 = src_index1 + (width - 1) * channels;
							for (int col = left + width; col < dst_width; ++col)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + src_index2 + ch];
								}
							}
						}
					}
				}
				else
				{
					LOG(ERROR) << "Un-support border type.";
					return;
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			dst = std::make_shared<memory::tensor<Dtype>>(dst_temp->clone());
		}
#endif

	}
}
#endif // !_OPERATION_MAKE_BORDER_HPP_
