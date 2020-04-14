#include "conv_1x1s1_cpu.hpp"
#include "../../include/Excalibur/tensor_operation_cpu.hpp"
#include <iostream>
#include <fstream>

using namespace glasssix::memory;

namespace glasssix
{
	namespace excalibur
	{
		conv_1x1s1_cpu::conv_1x1s1_cpu(int input_Channel, int output_Channel, int group, int kernelSize, int stride, int pad, bool bias_term, int device, bool int8_quantization)
			: baseconv(input_Channel, output_Channel, group, kernelSize, stride, pad, bias_term, device, int8_quantization)
		{

		}

		void conv_1x1s1_cpu::forward_gemm(const signed char* input, const signed char* weights, int* output, bool skip_im2col) {}

		void conv_1x1s1_cpu::forward_gemm(const float* input, const float* weights, float* output, bool skip_im2col) {}

		void conv_1x1s1_cpu::forward_bias(float* output, const float* bias) {}

		void conv_1x1s1_cpu::Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			CHECK_EQ(kernelSize_, 1);
			CHECK_EQ(stride_, 1);

			num_ = bottom->data_shape()[0];
			order_ = bottom->order();
			input_shape_.clear();
			input_shape_ = bottom->data_shape();
			bottom_dim_ = bottom->count(1, 4);

			int num = bottom->num();
			int w = bottom->width();
			int h = bottom->height();
			int inch = bottom->channels();
			int bottom_cstep = w * h;
			const int size = w * h;
			const float* bottom_data = bottom->cpu_data();

			int outch = output_Channel_;
			int outh = (h + 2 * pad_ - kernelSize_) / stride_ + 1;
			int outw = (w + 2 * pad_ - kernelSize_) / stride_ + 1;
			top.reset(new tensor<float>(std::vector<int>{num, outch, outh, outw}));
			int top_cstep = outw * outh;
			float* top_data = top->mutable_cpu_data();


			if (!int8_quantization_)
			{
			    int kernel_cstep = weights1x1_->width() * weights1x1_->height();
			    
			    // interleave
			    tensor<float> tmp(std::vector<int>{1, size / 8 + (size % 8) / 4 + size % 4, inch / 4 + inch % 4, 8 * 4}, -1, NCHW);
			    float *tmp_data = tmp.mutable_cpu_data();
			    int tmp_cstep = tmp.width() * tmp.height();
			    
			    std::shared_ptr<tensor<float>> bottom_bordered;
			    if (pad_ > 0)
			    {
			    	tensor_operation_cpu::make_border_cpu(bottom, bottom_bordered, pad_, pad_, pad_, pad_);
			    	if (bottom->order() == NHWC)
			    	{
			    		tensor_operation_cpu::nhwc2nchw_cpu(bottom_bordered, bottom_bordered);
			    	}
			    	h = bottom_bordered->height();
			    	w = bottom_bordered->width();
			    	bottom_cstep = w * h;
			    	bottom_data = bottom_bordered->cpu_data();
			    }
			    
			    for (int num_i = 0; num_i < num; num_i++)
			    {
			    	const float *bottom_data_num = bottom_data + num_i * inch * bottom_cstep;
			    	float *top_data_num = top_data + num_i * outch * top_cstep;
			    
			    	{
			    		int nn_size = size >> 3;
			    		int remain_size_start = nn_size << 3;
			    
#ifdef _OPENMP
#pragma omp parallel for
#endif		    
			    		for (int ii = 0; ii < nn_size; ii++)
			    		{
			    			int i = ii * 8;
			    
			    			const float* img0 = bottom_data_num;
			    			img0 += i;
			    
			    			float* tmpptr = tmp_data + (i / 8) * tmp_cstep;
			    
			    			for (int q = 0; q < inch; q++)
			    			{
#if SIMD_TYPE >= SIMDTYPE_AVX
							mm_type img_data = mm_load_ps(img0);
							mm_store_ps(tmpptr, img_data);
#elif SIMD_TYPE >= SIMDTYPE_SSE
			    				mm_type img_data = mm_load_ps(img0);
			    				mm_store_ps(tmpptr, img_data);
			    				img_data = mm_load_ps(img0 + mm_align_size);
			    				mm_store_ps(tmpptr + mm_align_size, img_data);
#else		    
			    				tmpptr[0] = img0[0];
			    				tmpptr[1] = img0[1];
			    				tmpptr[2] = img0[2];
			    				tmpptr[3] = img0[3];
			    				tmpptr[4] = img0[4];
			    				tmpptr[5] = img0[5];
			    				tmpptr[6] = img0[6];
			    				tmpptr[7] = img0[7];
#endif		    
			    				tmpptr += 8;
			    				img0 += bottom_cstep;
			    			}
			    		}
			    
			    		nn_size = (size - remain_size_start) >> 2;
			    
#ifdef _OPENMP
#pragma omp parallel for
#endif		    
			    		for (int ii = 0; ii < nn_size; ii++)
			    		{
			    			int i = remain_size_start + ii * 4;
			    
			    			const float* img0 = bottom_data_num;
			    			img0 += i;
			    
			    			float* tmpptr = tmp_data + (i / 8 + (i % 8) / 4) * tmp_cstep;
			    
			    			for (int q = 0; q < inch; q++)
			    			{
#if SIMD_TYPE >= SIMDTYPE_SSE
			    				__m128 img_data = _mm_loadu_ps(img0);
			    				_mm_storeu_ps(tmpptr, img_data);
#else		    
			    				tmpptr[0] = img0[0];
			    				tmpptr[1] = img0[1];
			    				tmpptr[2] = img0[2];
			    				tmpptr[3] = img0[3];
#endif		    
			    				tmpptr += 4;
			    				img0 += bottom_cstep;
			    			}
			    		}
			    
			    		remain_size_start += nn_size << 2;
			    
#ifdef _OPENMP
#pragma omp parallel for
#endif		    
			    		for (int i = remain_size_start; i < size; i++)
			    		{
			    			const float* img0 = bottom_data_num;
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
			    
			    	nn_outch = (outch - remain_outch_start) >> 2;
			    
#ifdef _OPENMP
#pragma omp parallel for
#endif		    
			    	for (int pp = 0; pp < nn_outch; pp++)
			    	{
			    		int p = remain_outch_start + pp * 4;
			    
			    		float* outptr0 = top_data_num + (p + 0) * top_cstep;
			    		float* outptr1 = top_data_num + (p + 1) * top_cstep;
			    		float* outptr2 = top_data_num + (p + 2) * top_cstep;
			    		float* outptr3 = top_data_num + (p + 3) * top_cstep;
			    
			    		const float zeros[4] = { 0.f, 0.f, 0.f, 0.f };
			    		const float* biasptr = bias_term_ ? bias_data + p : nullptr;
			    
			    		int i = 0;
			    
			    		for (; i + 7 < size; i += 8)
			    		{
			    			const float* tmpptr = tmp_data + (i / 8) * tmp_cstep;
			    			const float* kptr = weights1x1_data + (p / 4) * kernel_cstep;
			    
#if SIMD_TYPE >= SIMDTYPE_AVX
			    			mm_type sum0, sum1, sum2, sum3;
			    			if (bias_term_)
			    			{
			    				sum0 = mm_set1_ps(biasptr[0]);
			    				sum1 = mm_set1_ps(biasptr[1]);
			    				sum2 = mm_set1_ps(biasptr[2]);
			    				sum3 = mm_set1_ps(biasptr[3]);
			    			}
			    			else
			    			{
			    				sum0 = mm_setzero_ps();
			    				sum1 = mm_setzero_ps();
			    				sum2 = mm_setzero_ps();
			    				sum3 = mm_setzero_ps();
			    			}
			    
			    			for (int q = 0; q < inch; q++)
			    			{
			    				mm_type kptr0 = mm_set1_ps(kptr[0]);
			    				mm_type kptr1 = mm_set1_ps(kptr[1]);
			    				mm_type kptr2 = mm_set1_ps(kptr[2]);
			    				mm_type kptr3 = mm_set1_ps(kptr[3]);
			    
			    				mm_type tmpptr0 = mm_load_ps(tmpptr);
			    
			    				sum0 = mm_fmadd_ps(kptr0, tmpptr0, sum0);
			    				sum1 = mm_fmadd_ps(kptr1, tmpptr0, sum1);
			    				sum2 = mm_fmadd_ps(kptr2, tmpptr0, sum2);
			    				sum3 = mm_fmadd_ps(kptr3, tmpptr0, sum3);
			    
			    				tmpptr += 8;
			    				kptr += 4;
			    			}
			    
			    			mm_store_ps(outptr0, sum0);
			    			mm_store_ps(outptr1, sum1);
			    			mm_store_ps(outptr2, sum2);
			    			mm_store_ps(outptr3, sum3);
			    
			    			outptr0 += 8;
			    			outptr1 += 8;
			    			outptr2 += 8;
			    			outptr3 += 8;
#elif SIMD_TYPE >= SIMDTYPE_SSE
						mm_type sum0_0, sum0_4, sum1_0, sum1_4, sum2_0, sum2_4, sum3_0, sum3_4;
						if (bias_term_)
						{
							sum0_0 = mm_set1_ps(biasptr[0]);
							sum0_4 = mm_set1_ps(biasptr[0]);
							sum1_0 = mm_set1_ps(biasptr[1]);
							sum1_4 = mm_set1_ps(biasptr[1]);
							sum2_0 = mm_set1_ps(biasptr[2]);
							sum2_4 = mm_set1_ps(biasptr[2]);
							sum3_0 = mm_set1_ps(biasptr[3]);
							sum3_4 = mm_set1_ps(biasptr[3]);
						}
						else
						{
							sum0_0 = mm_setzero_ps();
							sum0_4 = mm_setzero_ps();
							sum1_0 = mm_setzero_ps();
							sum1_4 = mm_setzero_ps();
							sum2_0 = mm_setzero_ps();
							sum2_4 = mm_setzero_ps();
							sum3_0 = mm_setzero_ps();
							sum3_4 = mm_setzero_ps();
						}

						for (int q = 0; q < inch; q++)
						{
							mm_type kptr0 = mm_set1_ps(kptr[0]);
							mm_type kptr1 = mm_set1_ps(kptr[1]);
							mm_type kptr2 = mm_set1_ps(kptr[2]);
							mm_type kptr3 = mm_set1_ps(kptr[3]);

							mm_type tmpptr0_0 = mm_load_ps(tmpptr);
							mm_type tmpptr0_4 = mm_load_ps(tmpptr + 4);

							sum0_0 = mm_fmadd_ps(kptr0, tmpptr0_0, sum0_0);
							sum0_4 = mm_fmadd_ps(kptr0, tmpptr0_4, sum0_4);
							sum1_0 = mm_fmadd_ps(kptr1, tmpptr0_0, sum1_0);
							sum1_4 = mm_fmadd_ps(kptr1, tmpptr0_4, sum1_4);
							sum2_0 = mm_fmadd_ps(kptr2, tmpptr0_0, sum2_0);
							sum2_4 = mm_fmadd_ps(kptr2, tmpptr0_4, sum2_4);
							sum3_0 = mm_fmadd_ps(kptr3, tmpptr0_0, sum3_0);
							sum3_4 = mm_fmadd_ps(kptr3, tmpptr0_4, sum3_4);

							tmpptr += 8;
							kptr += 4;
						}

						mm_store_ps(outptr0, sum0_0);
						mm_store_ps(outptr0 + 4, sum0_4);
						mm_store_ps(outptr1, sum1_0);
						mm_store_ps(outptr1 + 4, sum1_4);
						mm_store_ps(outptr2, sum2_0);
						mm_store_ps(outptr2 + 4, sum2_4);
						mm_store_ps(outptr3, sum3_0);
						mm_store_ps(outptr3 + 4, sum3_4);

						outptr0 += 8;
						outptr1 += 8;
						outptr2 += 8;
						outptr3 += 8;
#else

                        float sum0_0, sum0_1, sum0_2, sum0_3, sum0_4, sum0_5, sum0_6, sum0_7;
						float sum1_0, sum1_1, sum1_2, sum1_3, sum1_4, sum1_5, sum1_6, sum1_7;
						float sum2_0, sum2_1, sum2_2, sum2_3, sum2_4, sum2_5, sum2_6, sum2_7;
						float sum3_0, sum3_1, sum3_2, sum3_3, sum3_4, sum3_5, sum3_6, sum3_7;

						if (bias_term_)
						{
							sum0_0 = biasptr[0];
							sum0_1 = biasptr[0];
							sum0_2 = biasptr[0];
							sum0_3 = biasptr[0];
							sum0_4 = biasptr[0];
							sum0_5 = biasptr[0];
							sum0_6 = biasptr[0];
							sum0_7 = biasptr[0];

							sum1_0 = biasptr[1];
							sum1_1 = biasptr[1];
							sum1_2 = biasptr[1];
							sum1_3 = biasptr[1];
							sum1_4 = biasptr[1];
							sum1_5 = biasptr[1];
							sum1_6 = biasptr[1];
							sum1_7 = biasptr[1];

							sum2_0 = biasptr[2];
							sum2_1 = biasptr[2];
							sum2_2 = biasptr[2];
							sum2_3 = biasptr[2];
							sum2_4 = biasptr[2];
							sum2_5 = biasptr[2];
							sum2_6 = biasptr[2];
							sum2_7 = biasptr[2];

							sum3_0 = biasptr[3];
							sum3_1 = biasptr[3];
							sum3_2 = biasptr[3];
							sum3_3 = biasptr[3];
							sum3_4 = biasptr[3];
							sum3_5 = biasptr[3];
							sum3_6 = biasptr[3];
							sum3_7 = biasptr[3];
						}
						else
						{
							sum0_0 = 0;
							sum0_1 = 0;
							sum0_2 = 0;
							sum0_3 = 0;
							sum0_4 = 0;
							sum0_5 = 0;
							sum0_6 = 0;
							sum0_7 = 0;
									 
							sum1_0 = 0;
							sum1_1 = 0;
							sum1_2 = 0;
							sum1_3 = 0;
							sum1_4 = 0;
							sum1_5 = 0;
							sum1_6 = 0;
							sum1_7 = 0;
									 
							sum2_0 = 0;
							sum2_1 = 0;
							sum2_2 = 0;
							sum2_3 = 0;
							sum2_4 = 0;
							sum2_5 = 0;
							sum2_6 = 0;
							sum2_7 = 0;
									 
							sum3_0 = 0;
							sum3_1 = 0;
							sum3_2 = 0;
							sum3_3 = 0;
							sum3_4 = 0;
							sum3_5 = 0;
							sum3_6 = 0;
							sum3_7 = 0;
						}						

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
#endif		    
			    		}
			    
			    		for (; i + 3 < size; i += 4)
			    		{
			    			const float* tmpptr = tmp_data + (i / 8 + (i % 8) / 4) * tmp_cstep;
			    			const float* kptr = weights1x1_data + (p / 4) * kernel_cstep;
			    
#if SIMD_TYPE >= SIMDTYPE_SSE
			    			__m128 sum0, sum1, sum2, sum3;
			    			if (bias_term_)
			    			{
			    				sum0 = _mm_set1_ps(biasptr[0]);
			    				sum1 = _mm_set1_ps(biasptr[1]);
			    				sum2 = _mm_set1_ps(biasptr[2]);
			    				sum3 = _mm_set1_ps(biasptr[3]);
			    			}
			    			else
			    			{
			    				sum0 = _mm_setzero_ps();
			    				sum1 = _mm_setzero_ps();
			    				sum2 = _mm_setzero_ps();
			    				sum3 = _mm_setzero_ps();
			    			}
			    
			    			for (int q = 0; q < inch; q++)
			    			{
			    				__m128 kptr0 = _mm_set1_ps(kptr[0]);
			    				__m128 kptr1 = _mm_set1_ps(kptr[1]);
			    				__m128 kptr2 = _mm_set1_ps(kptr[2]);
			    				__m128 kptr3 = _mm_set1_ps(kptr[3]);
			    				__m128 tmpptr0 = _mm_loadu_ps(tmpptr);
			    
#if USE_FMADD128
			    				sum0 = _mm_fmadd_ps(tmpptr0, kptr0, sum0);
			    				sum1 = _mm_fmadd_ps(tmpptr0, kptr1, sum1);
			    				sum2 = _mm_fmadd_ps(tmpptr0, kptr2, sum2);
			    				sum3 = _mm_fmadd_ps(tmpptr0, kptr3, sum3);
#else		    
			    				sum0 = _mm_add_ps(_mm_mul_ps(tmpptr0, kptr0), sum0);
			    				sum1 = _mm_add_ps(_mm_mul_ps(tmpptr0, kptr1), sum1);
			    				sum2 = _mm_add_ps(_mm_mul_ps(tmpptr0, kptr2), sum2);
			    				sum3 = _mm_add_ps(_mm_mul_ps(tmpptr0, kptr3), sum3);
#endif		    
			    				tmpptr += 4;
			    				kptr += 4;
			    			}
			    
			    			_mm_storeu_ps(outptr0, sum0);
			    			_mm_storeu_ps(outptr1, sum1);
			    			_mm_storeu_ps(outptr2, sum2);
			    			_mm_storeu_ps(outptr3, sum3);
			    
			    			outptr0 += 4;
			    			outptr1 += 4;
			    			outptr2 += 4;
			    			outptr3 += 4;
			    
#else		    
			    			float sum0_0, sum0_1, sum0_2, sum0_3;
			    			float sum1_0, sum1_1, sum1_2, sum1_3;
			    			float sum2_0, sum2_1, sum2_2, sum2_3;
			    			float sum3_0, sum3_1, sum3_2, sum3_3;
			    
			    			if (bias_term_)
			    			{
			    				sum0_0 = biasptr[0];
			    				sum0_1 = biasptr[0];
			    				sum0_2 = biasptr[0];
			    				sum0_3 = biasptr[0];
			    
			    				sum1_0 = biasptr[1];
			    				sum1_1 = biasptr[1];
			    				sum1_2 = biasptr[1];
			    				sum1_3 = biasptr[1];
			    
			    				sum2_0 = biasptr[2];
			    				sum2_1 = biasptr[2];
			    				sum2_2 = biasptr[2];
			    				sum2_3 = biasptr[2];
			    
			    				sum3_0 = biasptr[3];
			    				sum3_1 = biasptr[3];
			    				sum3_2 = biasptr[3];
			    				sum3_3 = biasptr[3];
			    			}
			    			else
			    			{
			    				sum0_0 = 0;
			    				sum0_1 = 0;
			    				sum0_2 = 0;
			    				sum0_3 = 0;
			    						 
			    				sum1_0 = 0;
			    				sum1_1 = 0;
			    				sum1_2 = 0;
			    				sum1_3 = 0;
			    						 
			    				sum2_0 = 0;
			    				sum2_1 = 0;
			    				sum2_2 = 0;
			    				sum2_3 = 0;
			    						 
			    				sum3_0 = 0;
			    				sum3_1 = 0;
			    				sum3_2 = 0;
			    				sum3_3 = 0;
			    			}
			    
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
#endif		    
			    		}
			    
			    		for (; i < size; i++)
			    		{
			    			const float* tmpptr = tmp_data + (i / 8 + (i % 8) / 4 + i % 4) * tmp_cstep;
			    			const float* kptr = weights1x1_data + (p / 4) * kernel_cstep;
			    
#if SIMD_TYPE >= SIMDTYPE_SSE
			    			__m128 sum = bias_term_ ? _mm_loadu_ps(biasptr) : _mm_setzero_ps();
			    
			    			for (int q = 0; q < inch; q++)
			    			{
			    				__m128 kptr0 = _mm_loadu_ps(kptr);
			    				__m128 tmpptr0 = _mm_set1_ps(tmpptr[0]);
			    
#if USE_FMADD128
			    				sum = _mm_fmadd_ps(tmpptr0, kptr0, sum);
#else		    
			    				sum = _mm_add_ps(_mm_mul_ps(tmpptr0, kptr0), sum);
#endif		    
			    				tmpptr++;
			    				kptr += 4;
			    			}
			    
			    			std::shared_ptr<tensor<float>> out;
			    			out.reset(new tensor<float>(std::vector<int>{4}));
			    			float *out_data = out->mutable_cpu_data();
			    			_mm_storeu_ps(out_data, sum);
			    
			    			outptr0[0] = out_data[0];
			    			outptr1[0] = out_data[1];
			    			outptr2[0] = out_data[2];
			    			outptr3[0] = out_data[3];
			    
			    			outptr0++;
			    			outptr1++;
			    			outptr2++;
			    			outptr3++;
#else		    
			    			float sum0, sum1, sum2, sum3;
			    			if (bias_term_)
			    			{
			    				sum0 = biasptr[0];
			    				sum1 = biasptr[1];
			    				sum2 = biasptr[2];
			    				sum3 = biasptr[3];
			    			}
			    			else
			    			{
			    				sum0 = 0;
			    				sum1 = 0;
			    				sum2 = 0;
			    				sum3 = 0;
			    			}
			    
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
#endif		    
			    		}
			    	}
			    
			    	remain_outch_start += nn_outch << 2;
			    
#ifdef _OPENMP
#pragma omp parallel for
#endif		    
			    	for (int p = remain_outch_start; p < outch; p++)
			    	{
			    		float* outptr0 = top_data_num + (p)* top_cstep;
			    
			    		const float bias0 = bias_term_ ? bias_data[p] : 0;
			    
			    		int i = 0;
			    
			    		for (; i + 7 < size; i += 8)
			    		{
			    			const float* tmpptr = tmp_data + (i / 8) * tmp_cstep;
			    			const float* kptr = weights1x1_data + (p / 4 + p % 4) * kernel_cstep;
			    
#if SIMD_TYPE >= SIMDTYPE_AVX
			    			mm_type sum = bias_term_ ? mm_set1_ps(bias0) : mm_setzero_ps();
			    			for (int q = 0; q < inch; q++)
			    			{
			    				mm_type tmpptr0 = mm_load_ps(tmpptr);
			    				mm_type kptr0 = mm_set1_ps(kptr[0]);
			    				sum = mm_fmadd_ps(tmpptr0, kptr0, sum);
			    				tmpptr += 8;
			    				kptr++;
			    			}
			    
			    			mm_store_ps(outptr0, sum);
			    			outptr0 += 8;
#elif SIMD_TYPE >= SIMDTYPE_SSE
			    			mm_type sum0, sum1;
			    			if (bias_term_)
			    			{
			    				sum0 = mm_set1_ps(bias0);
			    				sum1 = mm_set1_ps(bias0);
			    			}
			    			else
			    			{
			    				sum0 = mm_setzero_ps();
			    				sum1 = mm_setzero_ps();
			    			}
			    
			    			for (int q = 0; q < inch; q++)
			    			{
			    				mm_type tmpptr0 = mm_load_ps(tmpptr);
			    				mm_type tmpptr1 = mm_load_ps(tmpptr + 4);
			    				mm_type kptr0 = mm_set1_ps(kptr[0]);
			    				sum0 = mm_fmadd_ps(tmpptr0, kptr0, sum0);
			    				sum1 = mm_fmadd_ps(tmpptr1, kptr0, sum1);
			    				tmpptr += 8;
			    				kptr++;
			    			}
			    
			    			mm_store_ps(outptr0, sum0);
			    			mm_store_ps(outptr0 + 4, sum1);
			    			outptr0 += 8;
#else		    
			    			float sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;
			    			if (bias_term_)
			    			{
			    				sum0 = bias0;
			    				sum1 = bias0;
			    				sum2 = bias0;
			    				sum3 = bias0;
			    				sum4 = bias0;
			    				sum5 = bias0;
			    				sum6 = bias0;
			    				sum7 = bias0;
			    			}
			    			else
			    			{
			    				sum0 = 0;
			    				sum1 = 0;
			    				sum2 = 0;
			    				sum3 = 0;
			    				sum4 = 0;
			    				sum5 = 0;
			    				sum6 = 0;
			    				sum7 = 0;
			    			}
			    
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
#endif		    
			    		}
			    
			    		for (; i + 3 < size; i += 4)
			    		{
			    			const float* tmpptr = tmp_data + (i / 8 + (i % 8) / 4) * tmp_cstep;
			    			const float* kptr = weights1x1_data + (p / 4 + p % 4) * kernel_cstep;
			    
#if SIMD_TYPE >= SIMDTYPE_SSE
			    			__m128 sum = bias_term_ ? _mm_set1_ps(bias0) : _mm_setzero_ps();
			    
			    			for (int q = 0; q < inch; q++)
			    			{
			    				__m128 kptr0 = _mm_set1_ps(kptr[0]);
			    				__m128 tmpptr0 = _mm_loadu_ps(tmpptr);
			    
#if USE_FMADD128
			    				sum = _mm_fmadd_ps(tmpptr0, kptr0, sum);
#else		    
			    				sum = _mm_add_ps(_mm_mul_ps(tmpptr0, kptr0), sum);
#endif		    
			    
			    				tmpptr += 4;
			    				kptr++;
			    			}
			    
			    			_mm_storeu_ps(outptr0, sum);
			    			outptr0 += 4;
#else		    
			    			float sum0, sum1, sum2, sum3;
			    			if (bias_term_)
			    			{
			    				sum0 = bias0;
			    				sum1 = bias0;
			    				sum2 = bias0;
			    				sum3 = bias0;
			    			}
			    			else
			    			{
			    				sum0 = 0;
			    				sum1 = 0;
			    				sum2 = 0;
			    				sum3 = 0;
			    			}
			    
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
#endif		    
			    		}
			    
			    		for (; i < size; i++)
			    		{
			    			const float* tmpptr = tmp_data + (i / 8 + (i % 8) / 4 + i % 4) * tmp_cstep;
			    			const float* kptr = weights1x1_data + (p / 4 + p % 4) * tmp_cstep;
			    
			    			int q = 0;
			    
			    			float sum0 = bias_term_ ? bias0 : 0;
			    
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
			    } 
			}
			else
			{
			    float total_scale = scales_data[0] * scales_data[1];

    			top_int32_.reset(new tensor<int>(std::vector<int>{num, outch, outh, outw}));
    			top_int32_data = top_int32_->mutable_cpu_data();
    
    			bottom_int8_.reset(new tensor<signed char>(std::vector<int>{num_ * bottom_dim_}));
    			bottom_int8_data = bottom_int8_->mutable_cpu_data();
    
#if SIMD_TYPE >= SIMDTYPE_SSE
			    int circle_num = num_ * bottom_dim_ / mm_align_size;
			    mm_type scale = mm_set1_ps(scales_data[0]);
			    int index = 0;
			    
			    for (; index < circle_num; index++)
			    {
			    	int index_offset = index * mm_align_size;
			    	mm_type data = mm_load_ps(bottom_data + index_offset);
			    	mm_type res_mul = mm_mul_ps(data, scale);
			    	mm_type res_round = mm_round_ps(res_mul, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
			    	mm_store_ps(bottom_round_data_, res_round);
			    	for (int i = 0; i < mm_align_size; i++)
			    	{
			    		if (bottom_round_data_[i] > 127)
			    		{
			    			bottom_int8_data[index_offset + i] = (signed char)(127);
			    		}
			    		else if (bottom_round_data_[i] < -128)
			    		{
			    			bottom_int8_data[index_offset + i] = (signed char)(-128);
			    		}
			    		else
			    		{
			    			bottom_int8_data[index_offset + i] = (signed char)(bottom_round_data_[i]);
			    		}
			    	}
			    }
			    
			    for (index = mm_align_size * index; index < num_ * bottom_dim_; index++)
			    {
			    	bottom_int8_data[index] = float32_to_int8(bottom_data[index] * scales_data[0]);
			    }
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
			    for (int index = 0; index < num_ * bottom_dim_; index++)
			    {
			    	bottom_int8_data[index] = float32_to_int8(bottom_data[index] * scales_data[0]);
			    }
#endif

			    int kernel_cstep = weights1x1_int8_->width() * weights1x1_int8_->height();
			    
			    // interleave
			    tensor<signed char> tmp(std::vector<int>{1, size / 8 + (size % 8) / 4 + size % 4, inch / 4 + inch % 4, 8 * 4}, -1, NCHW);
				signed char *tmp_data = tmp.mutable_cpu_data();
			    int tmp_cstep = tmp.width() * tmp.height();
			    
				const signed char *bottom_const_int8_data = bottom_int8_->cpu_data();
			    std::shared_ptr<tensor<signed char>> bottom_bordered;
			    if (pad_ > 0)
			    {
			    	tensor_operation_cpu::make_border_cpu(bottom_int8_, bottom_bordered, pad_, pad_, pad_, pad_);
			    	if (bottom->order() == NHWC)
			    	{
			    		tensor_operation_cpu::nhwc2nchw_cpu(bottom_bordered, bottom_bordered);
			    	}
			    	h = bottom_bordered->height();
			    	w = bottom_bordered->width();
			    	bottom_cstep = w * h;
					bottom_const_int8_data = bottom_bordered->cpu_data();
			    }
			    
			    for (int num_i = 0; num_i < num; num_i++)
			    {
			    	const signed char *bottom_data_num = bottom_const_int8_data + num_i * inch * bottom_cstep;
			    	int *top_data_num = top_int32_data + num_i * outch * top_cstep;
					float *top_data_num_fp32 = top_data + num_i * outch * top_cstep;
			    
			    	{
			    		int nn_size = size >> 3;
			    		int remain_size_start = nn_size << 3;
			    
#ifdef _OPENMP
#pragma omp parallel for
#endif
					    for (int ii = 0; ii < nn_size; ii++)
					    {
					    	int i = ii * 8;
					    
					    	const signed char* img0 = bottom_data_num;
					    	img0 += i;
					    
							signed char* tmpptr = tmp_data + (i / 8) * tmp_cstep;
					    
					    	for (int q = 0; q < inch; q++)
					    	{
								memcpy(tmpptr, img0, 8 * sizeof(signed char));
				    			tmpptr += 8;
				    			img0 += bottom_cstep;
				    		}
				    	}
				    
				    	nn_size = (size - remain_size_start) >> 2;

#ifdef _OPENMP
#pragma omp parallel for
#endif
					    for (int ii = 0; ii < nn_size; ii++)
					    {
					    	int i = remain_size_start + ii * 4;
					    
					    	const signed char* img0 = bottom_data_num;
					    	img0 += i;
					    
					    	signed char* tmpptr = tmp_data + (i / 8 + (i % 8) / 4) * tmp_cstep;
					    
					    	for (int q = 0; q < inch; q++)
					    	{
								memcpy(tmpptr, img0, 4 * sizeof(signed char));
				    			tmpptr += 4;
				    			img0 += bottom_cstep;
				    		}
				    	}
				    
				    	remain_size_start += nn_size << 2;
				    
#ifdef _OPENMP
#pragma omp parallel for
#endif
				    	for (int i = remain_size_start; i < size; i++)
				    	{
				    		const signed char* img0 = bottom_data_num;
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
				    
				    int nn_outch = 0;
				    int remain_outch_start = 0;
				    
				    nn_outch = (outch - remain_outch_start) >> 2;

#ifdef _OPENMP
#pragma omp parallel for
#endif
			    	for (int pp = 0; pp < nn_outch; pp++)
			    	{
			    		int p = remain_outch_start + pp * 4;
			    
			    		int* outptr0 = top_data_num + (p + 0) * top_cstep;
			    		int* outptr1 = top_data_num + (p + 1) * top_cstep;
			    		int* outptr2 = top_data_num + (p + 2) * top_cstep;
			    		int* outptr3 = top_data_num + (p + 3) * top_cstep;

						float* outptr0_fp32 = top_data_num_fp32 + (p + 0) * top_cstep;
						float* outptr1_fp32 = top_data_num_fp32 + (p + 1) * top_cstep;
						float* outptr2_fp32 = top_data_num_fp32 + (p + 2) * top_cstep;
						float* outptr3_fp32 = top_data_num_fp32 + (p + 3) * top_cstep;
			    
			    		const float zeros[4] = { 0.f, 0.f, 0.f, 0.f };
			    		const float* biasptr = bias_term_ ? bias_data + p : nullptr;
			    
			    		int i = 0;
			    
			    		for (; i + 7 < size; i += 8)
			    		{
			    			const signed char* tmpptr = tmp_data + (i / 8) * tmp_cstep;
			    			const signed char* kptr = weights1x1_int8_data + (p / 4) * kernel_cstep;
			    
#if SIMD_TYPE >= SIMDTYPE_AVX
							__m256i sum0 = _mm256_setzero_si256();
							__m256i sum1 = _mm256_setzero_si256();
							__m256i sum2 = _mm256_setzero_si256();
							__m256i sum3 = _mm256_setzero_si256();
#endif

							for (int q = 0; q < inch; q++)
							{
#if SIMD_TYPE >= SIMDTYPE_AVX
								__m128i sum = _mm_setzero_si128();
								__m128i a = _mm_load_si128((__m128i*)tmpptr);
								__m128i a_int16 = _mm_cvtepi8_epi16(a);

								__m128i b = _mm_set1_epi8(kptr[0]);
                                __m128i b_int16 = _mm_cvtepi8_epi16(b);
								__m128i product_int16 = _mm_mullo_epi16(a_int16, b_int16);
								__m256i product_int32 = _mm256_cvtepi16_epi32(product_int16);
								sum0 = _mm256_add_epi32(product_int32, sum0);

								b = _mm_set1_epi8(kptr[1]);
								b_int16 = _mm_cvtepi8_epi16(b);
								product_int16 = _mm_mullo_epi16(a_int16, b_int16);
								product_int32 = _mm256_cvtepi16_epi32(product_int16);
								sum1 = _mm256_add_epi32(product_int32, sum1);

								b = _mm_set1_epi8(kptr[2]);
								b_int16 = _mm_cvtepi8_epi16(b);
								product_int16 = _mm_mullo_epi16(a_int16, b_int16);
								product_int32 = _mm256_cvtepi16_epi32(product_int16);
								sum2 = _mm256_add_epi32(product_int32, sum2);

								b = _mm_set1_epi8(kptr[3]);
								b_int16 = _mm_cvtepi8_epi16(b);
								product_int16 = _mm_mullo_epi16(a_int16, b_int16);
								product_int32 = _mm256_cvtepi16_epi32(product_int16);
								sum3 = _mm256_add_epi32(product_int32, sum3);
#else
								outptr0[0] += tmpptr[0] * kptr[0];
								outptr0[1] += tmpptr[1] * kptr[0];
								outptr0[2] += tmpptr[2] * kptr[0];
								outptr0[3] += tmpptr[3] * kptr[0];
								outptr0[4] += tmpptr[4] * kptr[0];
								outptr0[5] += tmpptr[5] * kptr[0];
								outptr0[6] += tmpptr[6] * kptr[0];
								outptr0[7] += tmpptr[7] * kptr[0];

								outptr1[0] += tmpptr[0] * kptr[1];
								outptr1[1] += tmpptr[1] * kptr[1];
								outptr1[2] += tmpptr[2] * kptr[1];
								outptr1[3] += tmpptr[3] * kptr[1];
								outptr1[4] += tmpptr[4] * kptr[1];
								outptr1[5] += tmpptr[5] * kptr[1];
								outptr1[6] += tmpptr[6] * kptr[1];
								outptr1[7] += tmpptr[7] * kptr[1];

								outptr2[0] += tmpptr[0] * kptr[2];
								outptr2[1] += tmpptr[1] * kptr[2];
								outptr2[2] += tmpptr[2] * kptr[2];
								outptr2[3] += tmpptr[3] * kptr[2];
								outptr2[4] += tmpptr[4] * kptr[2];
								outptr2[5] += tmpptr[5] * kptr[2];
								outptr2[6] += tmpptr[6] * kptr[2];
								outptr2[7] += tmpptr[7] * kptr[2];

								outptr3[0] += tmpptr[0] * kptr[3];
								outptr3[1] += tmpptr[1] * kptr[3];
								outptr3[2] += tmpptr[2] * kptr[3];
								outptr3[3] += tmpptr[3] * kptr[3];
								outptr3[4] += tmpptr[4] * kptr[3];
								outptr3[5] += tmpptr[5] * kptr[3];
								outptr3[6] += tmpptr[6] * kptr[3];
								outptr3[7] += tmpptr[7] * kptr[3];
#endif

								tmpptr += 8;
								kptr += 4;
							}

#if SIMD_TYPE >= SIMDTYPE_AVX
							_mm256_store_si256((__m256i*)outptr0, sum0);
							_mm256_store_si256((__m256i*)outptr1, sum1);
							_mm256_store_si256((__m256i*)outptr2, sum2);
							_mm256_store_si256((__m256i*)outptr3, sum3);
#endif

							if (bias_term_)
							{
								outptr0_fp32[0] = biasptr[0];
								outptr0_fp32[1] = biasptr[0];
								outptr0_fp32[2] = biasptr[0];
								outptr0_fp32[3] = biasptr[0];
								outptr0_fp32[4] = biasptr[0];
								outptr0_fp32[5] = biasptr[0];
								outptr0_fp32[6] = biasptr[0];
								outptr0_fp32[7] = biasptr[0];

								outptr1_fp32[0] = biasptr[1];
								outptr1_fp32[1] = biasptr[1];
								outptr1_fp32[2] = biasptr[1];
								outptr1_fp32[3] = biasptr[1];
								outptr1_fp32[4] = biasptr[1];
								outptr1_fp32[5] = biasptr[1];
								outptr1_fp32[6] = biasptr[1];
								outptr1_fp32[7] = biasptr[1];

								outptr2_fp32[0] = biasptr[2];
								outptr2_fp32[1] = biasptr[2];
								outptr2_fp32[2] = biasptr[2];
								outptr2_fp32[3] = biasptr[2];
								outptr2_fp32[4] = biasptr[2];
								outptr2_fp32[5] = biasptr[2];
								outptr2_fp32[6] = biasptr[2];
								outptr2_fp32[7] = biasptr[2];

								outptr3_fp32[0] = biasptr[3];
								outptr3_fp32[1] = biasptr[3];
								outptr3_fp32[2] = biasptr[3];
								outptr3_fp32[3] = biasptr[3];
								outptr3_fp32[4] = biasptr[3];
								outptr3_fp32[5] = biasptr[3];
								outptr3_fp32[6] = biasptr[3];
								outptr3_fp32[7] = biasptr[3];
							}
							
							outptr0_fp32[0] += outptr0[0] / total_scale;
							outptr0_fp32[1] += outptr0[1] / total_scale;
							outptr0_fp32[2] += outptr0[2] / total_scale;
							outptr0_fp32[3] += outptr0[3] / total_scale;
							outptr0_fp32[4] += outptr0[4] / total_scale;
							outptr0_fp32[5] += outptr0[5] / total_scale;
							outptr0_fp32[6] += outptr0[6] / total_scale;
							outptr0_fp32[7] += outptr0[7] / total_scale;

							outptr1_fp32[0] += outptr1[0] / total_scale;
							outptr1_fp32[1] += outptr1[1] / total_scale;
							outptr1_fp32[2] += outptr1[2] / total_scale;
							outptr1_fp32[3] += outptr1[3] / total_scale;
							outptr1_fp32[4] += outptr1[4] / total_scale;
							outptr1_fp32[5] += outptr1[5] / total_scale;
							outptr1_fp32[6] += outptr1[6] / total_scale;
							outptr1_fp32[7] += outptr1[7] / total_scale;

							outptr2_fp32[0] += outptr2[0] / total_scale;
							outptr2_fp32[1] += outptr2[1] / total_scale;
							outptr2_fp32[2] += outptr2[2] / total_scale;
							outptr2_fp32[3] += outptr2[3] / total_scale;
							outptr2_fp32[4] += outptr2[4] / total_scale;
							outptr2_fp32[5] += outptr2[5] / total_scale;
							outptr2_fp32[6] += outptr2[6] / total_scale;
							outptr2_fp32[7] += outptr2[7] / total_scale;

							outptr3_fp32[0] += outptr3[0] / total_scale;
							outptr3_fp32[1] += outptr3[1] / total_scale;
							outptr3_fp32[2] += outptr3[2] / total_scale;
							outptr3_fp32[3] += outptr3[3] / total_scale;
							outptr3_fp32[4] += outptr3[4] / total_scale;
							outptr3_fp32[5] += outptr3[5] / total_scale;
							outptr3_fp32[6] += outptr3[6] / total_scale;
							outptr3_fp32[7] += outptr3[7] / total_scale;							

							outptr0 += 8;
							outptr1 += 8;
							outptr2 += 8;
							outptr3 += 8;

							outptr0_fp32 += 8;
							outptr1_fp32 += 8;
							outptr2_fp32 += 8;
							outptr3_fp32 += 8;
			    		}
			    
			    		for (; i + 3 < size; i += 4)
			    		{
			    			const signed char* tmpptr = tmp_data + (i / 8 + (i % 8) / 4) * tmp_cstep;
			    			const signed char* kptr = weights1x1_int8_data + (p / 4) * kernel_cstep;			   

							for (int q = 0; q < inch; q++)
							{
								outptr0[0] += tmpptr[0] * kptr[0];
								outptr0[1] += tmpptr[1] * kptr[0];
								outptr0[2] += tmpptr[2] * kptr[0];
								outptr0[3] += tmpptr[3] * kptr[0];

								outptr1[0] += tmpptr[0] * kptr[1];
								outptr1[1] += tmpptr[1] * kptr[1];
								outptr1[2] += tmpptr[2] * kptr[1];
								outptr1[3] += tmpptr[3] * kptr[1];

								outptr2[0] += tmpptr[0] * kptr[2];
								outptr2[1] += tmpptr[1] * kptr[2];
								outptr2[2] += tmpptr[2] * kptr[2];
								outptr2[3] += tmpptr[3] * kptr[2];

								outptr3[0] += tmpptr[0] * kptr[3];
								outptr3[1] += tmpptr[1] * kptr[3];
								outptr3[2] += tmpptr[2] * kptr[3];
								outptr3[3] += tmpptr[3] * kptr[3];

								tmpptr += 4;
								kptr += 4;
						    }

							if (bias_term_)
							{
								outptr0_fp32[0] = biasptr[0];
								outptr0_fp32[1] = biasptr[0];
								outptr0_fp32[2] = biasptr[0];
								outptr0_fp32[3] = biasptr[0];

								outptr1_fp32[0] = biasptr[1];
								outptr1_fp32[1] = biasptr[1];
								outptr1_fp32[2] = biasptr[1];
								outptr1_fp32[3] = biasptr[1];

								outptr2_fp32[0] = biasptr[2];
								outptr2_fp32[1] = biasptr[2];
								outptr2_fp32[2] = biasptr[2];
								outptr2_fp32[3] = biasptr[2];

								outptr3_fp32[0] = biasptr[3];
								outptr3_fp32[1] = biasptr[3];
								outptr3_fp32[2] = biasptr[3];
								outptr3_fp32[3] = biasptr[3];
							}

							outptr0_fp32[0] += outptr0[0] / total_scale;
							outptr0_fp32[1] += outptr0[1] / total_scale;
							outptr0_fp32[2] += outptr0[2] / total_scale;
							outptr0_fp32[3] += outptr0[3] / total_scale;

							outptr1_fp32[0] += outptr1[0] / total_scale;
							outptr1_fp32[1] += outptr1[1] / total_scale;
							outptr1_fp32[2] += outptr1[2] / total_scale;
							outptr1_fp32[3] += outptr1[3] / total_scale;

							outptr2_fp32[0] += outptr2[0] / total_scale;
							outptr2_fp32[1] += outptr2[1] / total_scale;
							outptr2_fp32[2] += outptr2[2] / total_scale;
							outptr2_fp32[3] += outptr2[3] / total_scale;

							outptr3_fp32[0] += outptr3[0] / total_scale;
							outptr3_fp32[1] += outptr3[1] / total_scale;
							outptr3_fp32[2] += outptr3[2] / total_scale;
							outptr3_fp32[3] += outptr3[3] / total_scale;

							outptr0 += 4;
							outptr1 += 4;
							outptr2 += 4;
							outptr3 += 4;

							outptr0_fp32 += 4;
							outptr1_fp32 += 4;
							outptr2_fp32 += 4;
							outptr3_fp32 += 4;
			    		}
			    
			    		for (; i < size; i++)
			    		{
			    			const signed char* tmpptr = tmp_data + (i / 8 + (i % 8) / 4 + i % 4) * tmp_cstep;
			    			const signed char* kptr = weights1x1_int8_data + (p / 4) * kernel_cstep;
			    
							for (int q = 0; q < inch; q++)
							{
								outptr0[0] += tmpptr[0] * kptr[0];
								outptr1[0] += tmpptr[0] * kptr[1];
								outptr2[0] += tmpptr[0] * kptr[2];
								outptr3[0] += tmpptr[0] * kptr[3];

								tmpptr++;
								kptr += 4;
							}

							if (bias_term_)
							{
								outptr0_fp32[0] = biasptr[0];
								outptr1_fp32[0] = biasptr[1];
								outptr2_fp32[0] = biasptr[2];
								outptr3_fp32[0] = biasptr[3];
							}

							outptr0_fp32[0] += outptr0[0] / total_scale;
							outptr1_fp32[0] += outptr1[0] / total_scale;
							outptr2_fp32[0] += outptr2[0] / total_scale;
							outptr3_fp32[0] += outptr3[0] / total_scale;

							outptr0++;
							outptr1++;
							outptr2++;
							outptr3++;

							outptr0_fp32++;
							outptr1_fp32++;
							outptr2_fp32++;
							outptr3_fp32++;
			    		}
			    	}
			    
			    	remain_outch_start += nn_outch << 2;
			    
#ifdef _OPENMP
#pragma omp parallel for
#endif
		    		for (int p = remain_outch_start; p < outch; p++)
		    		{
		    			int* outptr0 = top_data_num + (p)* top_cstep;
						float* outptr0_fp32 = top_data_num_fp32 + (p)* top_cstep;
		    
		    			const float bias0 = bias_term_ ? bias_data[p] : 0;
		    
		    			int i = 0;
		    
		    			for (; i + 7 < size; i += 8)
		    			{
		    				const signed char* tmpptr = tmp_data + (i / 8) * tmp_cstep;
		    				const signed char* kptr = weights1x1_int8_data + (p / 4 + p % 4) * kernel_cstep;
		    
							for (int q = 0; q < inch; q++)
							{
								outptr0[0] += tmpptr[0] * kptr[0];
								outptr0[1] += tmpptr[1] * kptr[0];
								outptr0[2] += tmpptr[2] * kptr[0];
								outptr0[3] += tmpptr[3] * kptr[0];
								outptr0[4] += tmpptr[4] * kptr[0];
								outptr0[5] += tmpptr[5] * kptr[0];
								outptr0[6] += tmpptr[6] * kptr[0];
								outptr0[7] += tmpptr[7] * kptr[0];

								tmpptr += 8;
								kptr++;
							}

							if (bias_term_)
							{
								outptr0_fp32[0] = bias0;
								outptr0_fp32[1] = bias0;
								outptr0_fp32[2] = bias0;
								outptr0_fp32[3] = bias0;
								outptr0_fp32[4] = bias0;
								outptr0_fp32[5] = bias0;
								outptr0_fp32[6] = bias0;
								outptr0_fp32[7] = bias0;
							}

							outptr0_fp32[0] += outptr0[0] / total_scale;
							outptr0_fp32[1] += outptr0[1] / total_scale;
							outptr0_fp32[2] += outptr0[2] / total_scale;
							outptr0_fp32[3] += outptr0[3] / total_scale;
							outptr0_fp32[4] += outptr0[4] / total_scale;
							outptr0_fp32[5] += outptr0[5] / total_scale;
							outptr0_fp32[6] += outptr0[6] / total_scale;
							outptr0_fp32[7] += outptr0[7] / total_scale;

							outptr0 += 8;
							outptr0_fp32 += 8;
			    		}
			    
			    		for (; i + 3 < size; i += 4)
			    		{
			    			const signed char* tmpptr = tmp_data + (i / 8 + (i % 8) / 4) * tmp_cstep;
			    			const signed char* kptr = weights1x1_int8_data + (p / 4 + p % 4) * kernel_cstep;			

							for (int q = 0; q < inch; q++)
							{
								outptr0[0] += tmpptr[0] * kptr[0];
								outptr0[1] += tmpptr[1] * kptr[0];
								outptr0[2] += tmpptr[2] * kptr[0];
								outptr0[3] += tmpptr[3] * kptr[0];

								tmpptr += 4;
								kptr++;
							}

							if (bias_term_)
							{
								outptr0_fp32[0] = bias0;
								outptr0_fp32[1] = bias0;
								outptr0_fp32[2] = bias0;
								outptr0_fp32[3] = bias0;
							}

							outptr0_fp32[0] += outptr0[0] / total_scale;
							outptr0_fp32[1] += outptr0[1] / total_scale;
							outptr0_fp32[2] += outptr0[2] / total_scale;
							outptr0_fp32[3] += outptr0[3] / total_scale;

							outptr0 += 4;
							outptr0_fp32 += 4;
		    			}
		    
		    			for (; i < size; i++)
		    			{
		    				const signed char* tmpptr = tmp_data + (i / 8 + (i % 8) / 4 + i % 4) * tmp_cstep;
		    				const signed char* kptr = weights1x1_int8_data + (p / 4 + p % 4) * tmp_cstep;
		    
		    				int q = 0;

		    				for (; q < inch; q++)
		    				{
								outptr0[0] += tmpptr[0] * kptr[0];
		    					tmpptr++;
		    					kptr++;
		    				}
		    
							outptr0_fp32[0] = bias_term_ ? bias0 : 0;
		    				outptr0_fp32[0] += outptr0[0] / total_scale;
		    
		    				outptr0++;
							outptr0_fp32++;
		    			}
		    		}
		    	}
			}
		}
	}
}