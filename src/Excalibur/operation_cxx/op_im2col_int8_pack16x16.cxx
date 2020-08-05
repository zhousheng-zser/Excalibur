		template <typename Dtype>
		void operation_convolution<Dtype>::conv_im2col_sgemm_int8_dequant_sse(const std::shared_ptr<memory::tensor<signed char> >& bottom_blob, 
			std::shared_ptr<memory::tensor<float> >& top_blob, float scale_dequant)
		{
			int w = bottom_blob->width();
			int h = bottom_blob->height();
			int inch = bottom_blob->channels();
			int bottom_cstep = w * h;
			int outw = top_blob->width();
			int outh = top_blob->height();
			int outch = top_blob->channels();

			const signed char* kernel = weights_i8_[0]->cpu_data();
			const float* bias = nullptr;
			if (bias_term_)
				bias = weights_f32_[1]->cpu_data();

			float* top_data = top_blob->mutable_cpu_data();
			int top_cstep = top_blob->width() * top_blob->height();

			const signed char* bottom_data = bottom_blob->cpu_data();
			int out_size = outw * outh;
			int kernel_size = kernel_size_w_ * kernel_size_h_;
			int stride = kernel_size_h_ * kernel_size_w_ * outw * outh;

			// im2row
			bottom_im2row_.reset(new memory::tensor<signed char>(std::vector<int>{1, 1, kernel_size* inch,out_size }, params_.device_, memory::NCHW, bottom_blob->allocator()));
			signed char* ret = bottom_im2row_->mutable_cpu_data();
			{
#ifdef _OPENMP 
#pragma omp parallel for num_threads(2) 
#endif
				for (int p = 0; p < inch; p++)
				{
					const signed char* input = bottom_data + (p)*bottom_cstep;
					int retID = stride * p;
					for (int u = 0; u < kernel_size_h_; u++)
					{
						for (int v = 0; v < kernel_size_w_; v++)
						{
							for (int i = 0; i < outh; i++)
							{
								for (int j = 0; j < outw; j++)
								{
									int row = u + i * stride_h_;
									int col = v + j * stride_w_;
									int index = row * w + col;
									ret[retID] = input[index];
									retID++;
								}
							}
						}
					}
				}
			}

			const signed char* bottom_im2col_data = bottom_im2row_->cpu_data();

			// int M = outch;  // outch
			int N = outw * outh; // outsize or out stride
			int K = kernel_size_w_ * kernel_size_h_ * inch; // ksize * inch

			// bottom_im2row_ memory packed 16 x 16
			bottom_tm_int8_.reset(new memory::tensor<signed char>(std::vector<int>{1, out_size / 16 + out_size % 16, inch, 16 * kernel_size}, params_.device_, memory::NCHW, bottom_blob->allocator()));
			signed char* bottom_tm_data = bottom_tm_int8_->mutable_cpu_data();
			int bottom_tm_cstep = bottom_tm_int8_->width() * bottom_tm_int8_->height();
			{
				int nn_size = out_size >> 4;
				int remain_size_start = nn_size << 4;
				
#ifdef _OPENMP 
#pragma omp parallel for num_threads(2) 
#endif
				for (int ii = 0; ii < nn_size; ii++)
				{
					int i = ii * 16;

					const signed char* img0 = bottom_im2col_data;
					img0 += i;
					signed char* tmpptr = bottom_tm_data + (i / 16) * bottom_tm_cstep;

					for (int q = 0; q < inch * kernel_size; q++)
					{
#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX //AVX			
						_mm_storeu_si128((__m128i*)tmpptr, _mm_loadu_si128((__m128i*)img0));

#else					
						tmpptr[0] = img0[0];
						tmpptr[1] = img0[1];
						tmpptr[2] = img0[2];
						tmpptr[3] = img0[3];
						tmpptr[4] = img0[4];
						tmpptr[5] = img0[5];
						tmpptr[6] = img0[6];
						tmpptr[7] = img0[7];
						tmpptr[8] = img0[8];
						tmpptr[9] = img0[9];
						tmpptr[10] = img0[10];
						tmpptr[11] = img0[11];
						tmpptr[12] = img0[12];
						tmpptr[13] = img0[13];
						tmpptr[14] = img0[14];
						tmpptr[15] = img0[15];
#endif       
						tmpptr += 16;
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
					signed char* tmpptr = bottom_tm_data + (i / 16 + i % 16) * bottom_tm_cstep;
					for (int q = 0; q < inch * kernel_size; q++)
					{
						tmpptr[0] = img0[0];
						tmpptr += 1;
						img0 += out_size;
					}
				}
			}


			// kernel memory packed 16 x 16
			int kernel_tm_channel = outch / 16 + (outch % 16) / 8 + (outch % 8) / 4 + outch % 4;
			kernel_tm_int8_sgemm_.reset(new memory::tensor<signed char>(std::vector<int>{1, kernel_tm_channel, inch, 16 * kernel_size}, params_.device_, memory::NCHW, bottom_blob->allocator()));
			signed char* kernel_tm_data = kernel_tm_int8_sgemm_->mutable_cpu_data();
			int kernel_tm_cstep = kernel_tm_int8_sgemm_->width() * kernel_tm_int8_sgemm_->height();
			{
				int nn_outch = 0;
				int remain_outch_start = 0;

				nn_outch = outch >> 4;
				remain_outch_start = nn_outch << 4;

#ifdef _OPENMP 
#pragma omp parallel for num_threads(2) 
#endif
				for (int pp = 0; pp < nn_outch; pp++)
				{
					int p = pp * 16;

					const signed char* k0 = kernel + (p + 0) * inch * kernel_size;
					const signed char* k1 = kernel + (p + 1) * inch * kernel_size;
					const signed char* k2 = kernel + (p + 2) * inch * kernel_size;
					const signed char* k3 = kernel + (p + 3) * inch * kernel_size;
					const signed char* k4 = kernel + (p + 4) * inch * kernel_size;
					const signed char* k5 = kernel + (p + 5) * inch * kernel_size;
					const signed char* k6 = kernel + (p + 6) * inch * kernel_size;
					const signed char* k7 = kernel + (p + 7) * inch * kernel_size;
					const signed char* k8 = kernel + (p + 8) * inch * kernel_size;
					const signed char* k9 = kernel + (p + 9) * inch * kernel_size;
					const signed char* k10 = kernel + (p + 10) * inch * kernel_size;
					const signed char* k11 = kernel + (p + 11) * inch * kernel_size;
					const signed char* k12 = kernel + (p + 12) * inch * kernel_size;
					const signed char* k13 = kernel + (p + 13) * inch * kernel_size;
					const signed char* k14 = kernel + (p + 14) * inch * kernel_size;
					const signed char* k15 = kernel + (p + 15) * inch * kernel_size;

					signed char* ktmp = kernel_tm_data + (p / 16) * kernel_tm_cstep;
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
						ktmp[8] = k8[0];
						ktmp[9] = k9[0];
						ktmp[10] = k10[0];
						ktmp[11] = k11[0];
						ktmp[12] = k12[0];
						ktmp[13] = k13[0];
						ktmp[14] = k14[0];
						ktmp[15] = k15[0];
						ktmp += 16;

						k0 += 1; k1 += 1;
						k2 += 1; k3 += 1;
						k4 += 1; k5 += 1;
						k6 += 1; k7 += 1;
						k8 += 1; k9 += 1;
						k10 += 1; k11 += 1;
						k12 += 1; k13 += 1;
						k14 += 1; k15 += 1;
					}
				}

				nn_outch = (outch - remain_outch_start) >> 3;

				for (int pp = 0; pp < nn_outch; pp++)
				{
					int p = remain_outch_start + pp * 8;

					const signed char* k0 = kernel + (p + 0) * inch * kernel_size;
					const signed char* k1 = kernel + (p + 1) * inch * kernel_size;
					const signed char* k2 = kernel + (p + 2) * inch * kernel_size;
					const signed char* k3 = kernel + (p + 3) * inch * kernel_size;
					const signed char* k4 = kernel + (p + 4) * inch * kernel_size;
					const signed char* k5 = kernel + (p + 5) * inch * kernel_size;
					const signed char* k6 = kernel + (p + 6) * inch * kernel_size;
					const signed char* k7 = kernel + (p + 7) * inch * kernel_size;


					signed char* ktmp = kernel_tm_data + (p / 16 + (p % 16) / 8) * kernel_tm_cstep;
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

						k0 += 1; k1 += 1;
						k2 += 1; k3 += 1;
						k4 += 1; k5 += 1;
						k6 += 1; k7 += 1;
					}
				}
				remain_outch_start += nn_outch << 3;
				nn_outch = (outch - remain_outch_start) >> 2;
				for (int pp = 0; pp < nn_outch; pp++)
				{
					int p = remain_outch_start + pp * 4;

					const signed char* k0 = kernel + (p + 0) * inch * kernel_size;
					const signed char* k1 = kernel + (p + 1) * inch * kernel_size;
					const signed char* k2 = kernel + (p + 2) * inch * kernel_size;
					const signed char* k3 = kernel + (p + 3) * inch * kernel_size;


					signed char* ktmp = kernel_tm_data + (p / 16 + (p % 16) / 8 + (p % 8) / 4) * kernel_tm_cstep;
					for (int q = 0; q < inch * kernel_size; q++)
					{
						ktmp[0] = k0[0];
						ktmp[1] = k1[0];
						ktmp[2] = k2[0];
						ktmp[3] = k3[0];
						ktmp += 4;

						k0 += 1; k1 += 1;
						k2 += 1; k3 += 1;
					}
				}
				remain_outch_start += nn_outch << 2;
				for (int p = remain_outch_start; p < outch; p++)
				{
					const signed char* k0 = kernel + (p + 0) * inch * kernel_size;
					signed char* ktmp = kernel_tm_data + (p / 16 + (p % 16) / 8 + (p % 8) / 4 + p % 4) * kernel_tm_cstep;
					for (int q = 0; q < inch * kernel_size; q++)
					{
						ktmp[0] = k0[0];
						ktmp++;
						k0++;
					}
				}
			}

			{
				int nn_outch = 0;
				int remain_outch_start = 0;

				nn_outch = outch >> 4;
				remain_outch_start = nn_outch << 4;

#ifdef _OPENMP 
#pragma omp parallel for num_threads(2) 
#endif
				for (int pp = 0; pp < nn_outch; pp++)
				{
					int i = pp * 16;

					const float bias0 = bias ? bias[i] : 0.f;
					const float bias1 = bias ? bias[i + 1] : 0.f;
					const float bias2 = bias ? bias[i + 2] : 0.f;
					const float bias3 = bias ? bias[i + 3] : 0.f;
					const float bias4 = bias ? bias[i + 4] : 0.f;
					const float bias5 = bias ? bias[i + 5] : 0.f;
					const float bias6 = bias ? bias[i + 6] : 0.f;
					const float bias7 = bias ? bias[i + 7] : 0.f;
					const float bias8 = bias ? bias[i + 8] : 0.f;
					const float bias9 = bias ? bias[i + 9] : 0.f;
					const float bias10 = bias ? bias[i + 10] : 0.f;
					const float bias11 = bias ? bias[i + 11] : 0.f;
					const float bias12 = bias ? bias[i + 12] : 0.f;
					const float bias13 = bias ? bias[i + 13] : 0.f;
					const float bias14 = bias ? bias[i + 14] : 0.f;
					const float bias15 = bias ? bias[i + 15] : 0.f;

					const float scale_dequant0 = scale_dequant;

					float* output0 = top_data + (i)*top_cstep;
					float* output1 = top_data + (i + 1) * top_cstep;
					float* output2 = top_data + (i + 2) * top_cstep;
					float* output3 = top_data + (i + 3) * top_cstep;
					float* output4 = top_data + (i + 4) * top_cstep;
					float* output5 = top_data + (i + 5) * top_cstep;
					float* output6 = top_data + (i + 6) * top_cstep;
					float* output7 = top_data + (i + 7) * top_cstep;
					float* output8 = top_data + (i + 8) * top_cstep;
					float* output9 = top_data + (i + 9) * top_cstep;
					float* output10 = top_data + (i + 10) * top_cstep;
					float* output11 = top_data + (i + 11) * top_cstep;
					float* output12 = top_data + (i + 12) * top_cstep;
					float* output13 = top_data + (i + 13) * top_cstep;
					float* output14 = top_data + (i + 14) * top_cstep;
					float* output15 = top_data + (i + 15) * top_cstep;

					int j = 0;
					for (; j + 15 < N; j = j + 16)
					{
						signed char* vb = bottom_tm_data + (j / 16) * bottom_tm_cstep;
						signed char* va = kernel_tm_data + (i / 16) * kernel_tm_cstep;
						int sum0[16] = { 0 };
						int sum1[16] = { 0 };
						int sum2[16] = { 0 };
						int sum3[16] = { 0 };
						int sum4[16] = { 0 };
						int sum5[16] = { 0 };
						int sum6[16] = { 0 };
						int sum7[16] = { 0 };
						int sum8[16] = { 0 };
						int sum9[16] = { 0 };
						int sum10[16] = { 0 };
						int sum11[16] = { 0 };
						int sum12[16] = { 0 };
						int sum13[16] = { 0 };
						int sum14[16] = { 0 };
						int sum15[16] = { 0 };

#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX //AVX
						__m256i _sum0_0 = _mm256_setzero_si256();
						__m256i _sum0_1 = _mm256_setzero_si256();
						__m256i _sum1_0 = _mm256_setzero_si256();
						__m256i _sum1_1 = _mm256_setzero_si256();
						__m256i _sum2_0 = _mm256_setzero_si256();
						__m256i _sum2_1 = _mm256_setzero_si256();
						__m256i _sum3_0 = _mm256_setzero_si256();
						__m256i _sum3_1 = _mm256_setzero_si256();
						__m256i _sum4_0 = _mm256_setzero_si256();
						__m256i _sum4_1 = _mm256_setzero_si256();
						__m256i _sum5_0 = _mm256_setzero_si256();
						__m256i _sum5_1 = _mm256_setzero_si256();
						__m256i _sum6_0 = _mm256_setzero_si256();
						__m256i _sum6_1 = _mm256_setzero_si256();
						__m256i _sum7_0 = _mm256_setzero_si256();
						__m256i _sum7_1 = _mm256_setzero_si256();
						__m256i _sum8_0 = _mm256_setzero_si256();
						__m256i _sum8_1 = _mm256_setzero_si256();
						__m256i _sum9_0 = _mm256_setzero_si256();
						__m256i _sum9_1 = _mm256_setzero_si256();
						__m256i _sum10_0 = _mm256_setzero_si256();
						__m256i _sum10_1 = _mm256_setzero_si256();
						__m256i _sum11_0 = _mm256_setzero_si256();
						__m256i _sum11_1 = _mm256_setzero_si256();
						__m256i _sum12_0 = _mm256_setzero_si256();
						__m256i _sum12_1 = _mm256_setzero_si256();
						__m256i _sum13_0 = _mm256_setzero_si256();
						__m256i _sum13_1 = _mm256_setzero_si256();
						__m256i _sum14_0 = _mm256_setzero_si256();
						__m256i _sum14_1 = _mm256_setzero_si256();
						__m256i _sum15_0 = _mm256_setzero_si256();
						__m256i _sum15_1 = _mm256_setzero_si256();

						int k = 0;
						for (; k + 3 < K; k = k + 4)
						{
							__m128i va0 = _mm_set1_epi8(va[0]);
							__m128i va1 = _mm_set1_epi8(va[1]);
							__m128i va2 = _mm_set1_epi8(va[2]);
							__m128i va3 = _mm_set1_epi8(va[3]);
							__m128i va4 = _mm_set1_epi8(va[4]);
							__m128i va5 = _mm_set1_epi8(va[5]);
							__m128i va6 = _mm_set1_epi8(va[6]);
							__m128i va7 = _mm_set1_epi8(va[7]);
				

							__m128i vb0 = _mm_loadu_si128((__m128i*)vb);
							__m128i vb1 = _mm_loadu_si128((__m128i*)(vb + 16));
							__m128i vb2 = _mm_loadu_si128((__m128i*)(vb + 32));
							__m128i vb3 = _mm_loadu_si128((__m128i*)(vb + 48));
						    __m128i vb4 = _mm_loadu_si128((__m128i*)(vb + 64));
							__m128i vb5 = _mm_loadu_si128((__m128i*)(vb + 80));
							__m128i vb6 = _mm_loadu_si128((__m128i*)(vb + 96));
							__m128i vb7 = _mm_loadu_si128((__m128i*)(vb + 112));
							__m128i vb8 = _mm_loadu_si128((__m128i*)(vb + 128));
							__m128i vb9 = _mm_loadu_si128((__m128i*)(vb + 144));
							__m128i vb10 = _mm_loadu_si128((__m128i*)(vb + 160));
							__m128i vb11 = _mm_loadu_si128((__m128i*)(vb + 176));
							__m128i vb12 = _mm_loadu_si128((__m128i*)(vb + 192));
							__m128i vb13 = _mm_loadu_si128((__m128i*)(vb + 208));
							__m128i vb14 = _mm_loadu_si128((__m128i*)(vb + 224));
							__m128i vb15 = _mm_loadu_si128((__m128i*)(vb + 240));


							_mm256_epi16_epi32(vb0, va0, _sum0_0, _sum0_1);
							_mm256_epi16_epi32(vb0, va1, _sum1_0, _sum1_1);
							_mm256_epi16_epi32(vb0, va2, _sum2_0, _sum2_1);
							_mm256_epi16_epi32(vb0, va3, _sum3_0, _sum3_1);
							_mm256_epi16_epi32(vb0, va4, _sum4_0, _sum4_1);
							_mm256_epi16_epi32(vb0, va5, _sum5_0, _sum5_1);
							_mm256_epi16_epi32(vb0, va6, _sum6_0, _sum6_1);
							_mm256_epi16_epi32(vb0, va7, _sum7_0, _sum7_1);


							va0 = _mm_set1_epi8(va[8]);
							va1 = _mm_set1_epi8(va[9]);
							va2 = _mm_set1_epi8(va[10]);
							va3 = _mm_set1_epi8(va[11]);
							va4 = _mm_set1_epi8(va[12]);
							va5 = _mm_set1_epi8(va[13]);
							va6 = _mm_set1_epi8(va[14]);
							va7 = _mm_set1_epi8(va[15]);

							_mm256_epi16_epi32(vb0, va0, _sum8_0, _sum8_1);
							_mm256_epi16_epi32(vb0, va1, _sum9_0, _sum9_1);
							_mm256_epi16_epi32(vb0, va2, _sum10_0, _sum10_1);
							_mm256_epi16_epi32(vb0, va3, _sum11_0, _sum11_1);
							_mm256_epi16_epi32(vb0, va4, _sum12_0, _sum12_1);
							_mm256_epi16_epi32(vb0, va5, _sum13_0, _sum13_1);
							_mm256_epi16_epi32(vb0, va6, _sum14_0, _sum14_1);
							_mm256_epi16_epi32(vb0, va7, _sum15_0, _sum15_1);

							va += 16;



							va0 = _mm_set1_epi8(va[0]);
							va1 = _mm_set1_epi8(va[1]);
							va2 = _mm_set1_epi8(va[2]);
							va3 = _mm_set1_epi8(va[3]);
							va4 = _mm_set1_epi8(va[4]);
							va5 = _mm_set1_epi8(va[5]);
							va6 = _mm_set1_epi8(va[6]);
							va7 = _mm_set1_epi8(va[7]);


							_mm256_epi16_epi32(vb1, va0, _sum0_0, _sum0_1);
							_mm256_epi16_epi32(vb1, va1, _sum1_0, _sum1_1);
							_mm256_epi16_epi32(vb1, va2, _sum2_0, _sum2_1);
							_mm256_epi16_epi32(vb1, va3, _sum3_0, _sum3_1);
							_mm256_epi16_epi32(vb1, va4, _sum4_0, _sum4_1);
							_mm256_epi16_epi32(vb1, va5, _sum5_0, _sum5_1);
							_mm256_epi16_epi32(vb1, va6, _sum6_0, _sum6_1);
							_mm256_epi16_epi32(vb1, va7, _sum7_0, _sum7_1);


							va0 = _mm_set1_epi8(va[8]);
							va1 = _mm_set1_epi8(va[9]);
							va2 = _mm_set1_epi8(va[10]);
							va3 = _mm_set1_epi8(va[11]);
							va4 = _mm_set1_epi8(va[12]);
							va5 = _mm_set1_epi8(va[13]);
							va6 = _mm_set1_epi8(va[14]);
							va7 = _mm_set1_epi8(va[15]);

							_mm256_epi16_epi32(vb1, va0, _sum8_0, _sum8_1);
							_mm256_epi16_epi32(vb1, va1, _sum9_0, _sum9_1);
							_mm256_epi16_epi32(vb1, va2, _sum10_0, _sum10_1);
							_mm256_epi16_epi32(vb1, va3, _sum11_0, _sum11_1);
							_mm256_epi16_epi32(vb1, va4, _sum12_0, _sum12_1);
							_mm256_epi16_epi32(vb1, va5, _sum13_0, _sum13_1);
							_mm256_epi16_epi32(vb1, va6, _sum14_0, _sum14_1);
							_mm256_epi16_epi32(vb1, va7, _sum15_0, _sum15_1);

							va += 16;


							va0 = _mm_set1_epi8(va[0]);
							va1 = _mm_set1_epi8(va[1]);
							va2 = _mm_set1_epi8(va[2]);
							va3 = _mm_set1_epi8(va[3]);
							va4 = _mm_set1_epi8(va[4]);
							va5 = _mm_set1_epi8(va[5]);
							va6 = _mm_set1_epi8(va[6]);
							va7 = _mm_set1_epi8(va[7]);


							_mm256_epi16_epi32(vb2, va0, _sum0_0, _sum0_1);
							_mm256_epi16_epi32(vb2, va1, _sum1_0, _sum1_1);
							_mm256_epi16_epi32(vb2, va2, _sum2_0, _sum2_1);
							_mm256_epi16_epi32(vb2, va3, _sum3_0, _sum3_1);
							_mm256_epi16_epi32(vb2, va4, _sum4_0, _sum4_1);
							_mm256_epi16_epi32(vb2, va5, _sum5_0, _sum5_1);
							_mm256_epi16_epi32(vb2, va6, _sum6_0, _sum6_1);
							_mm256_epi16_epi32(vb2, va7, _sum7_0, _sum7_1);


							va0 = _mm_set1_epi8(va[8]);
							va1 = _mm_set1_epi8(va[9]);
							va2 = _mm_set1_epi8(va[10]);
							va3 = _mm_set1_epi8(va[11]);
							va4 = _mm_set1_epi8(va[12]);
							va5 = _mm_set1_epi8(va[13]);
							va6 = _mm_set1_epi8(va[14]);
							va7 = _mm_set1_epi8(va[15]);

							_mm256_epi16_epi32(vb2, va0, _sum8_0, _sum8_1);
							_mm256_epi16_epi32(vb2, va1, _sum9_0, _sum9_1);
							_mm256_epi16_epi32(vb2, va2, _sum10_0, _sum10_1);
							_mm256_epi16_epi32(vb2, va3, _sum11_0, _sum11_1);
							_mm256_epi16_epi32(vb2, va4, _sum12_0, _sum12_1);
							_mm256_epi16_epi32(vb2, va5, _sum13_0, _sum13_1);
							_mm256_epi16_epi32(vb2, va6, _sum14_0, _sum14_1);
							_mm256_epi16_epi32(vb2, va7, _sum15_0, _sum15_1);

							va += 16;

							va0 = _mm_set1_epi8(va[0]);
							va1 = _mm_set1_epi8(va[1]);
							va2 = _mm_set1_epi8(va[2]);
							va3 = _mm_set1_epi8(va[3]);
							va4 = _mm_set1_epi8(va[4]);
							va5 = _mm_set1_epi8(va[5]);
							va6 = _mm_set1_epi8(va[6]);
							va7 = _mm_set1_epi8(va[7]);


							_mm256_epi16_epi32(vb3, va0, _sum0_0, _sum0_1);
							_mm256_epi16_epi32(vb3, va1, _sum1_0, _sum1_1);
							_mm256_epi16_epi32(vb3, va2, _sum2_0, _sum2_1);
							_mm256_epi16_epi32(vb3, va3, _sum3_0, _sum3_1);
							_mm256_epi16_epi32(vb3, va4, _sum4_0, _sum4_1);
							_mm256_epi16_epi32(vb3, va5, _sum5_0, _sum5_1);
							_mm256_epi16_epi32(vb3, va6, _sum6_0, _sum6_1);
							_mm256_epi16_epi32(vb3, va7, _sum7_0, _sum7_1);


							va0 = _mm_set1_epi8(va[8]);
							va1 = _mm_set1_epi8(va[9]);
							va2 = _mm_set1_epi8(va[10]);
							va3 = _mm_set1_epi8(va[11]);
							va4 = _mm_set1_epi8(va[12]);
							va5 = _mm_set1_epi8(va[13]);
							va6 = _mm_set1_epi8(va[14]);
							va7 = _mm_set1_epi8(va[15]);

							_mm256_epi16_epi32(vb3, va0, _sum8_0, _sum8_1);
							_mm256_epi16_epi32(vb3, va1, _sum9_0, _sum9_1);
							_mm256_epi16_epi32(vb3, va2, _sum10_0, _sum10_1);
							_mm256_epi16_epi32(vb3, va3, _sum11_0, _sum11_1);
							_mm256_epi16_epi32(vb3, va4, _sum12_0, _sum12_1);
							_mm256_epi16_epi32(vb3, va5, _sum13_0, _sum13_1);
							_mm256_epi16_epi32(vb3, va6, _sum14_0, _sum14_1);
							_mm256_epi16_epi32(vb3, va7, _sum15_0, _sum15_1);

							va += 16;
							vb += 64;

						}
						for (; k < K; k++)
						{
							__m128i va0 = _mm_set1_epi8(va[0]);
							__m128i va1 = _mm_set1_epi8(va[1]);
							__m128i va2 = _mm_set1_epi8(va[2]);
							__m128i va3 = _mm_set1_epi8(va[3]);
							__m128i va4 = _mm_set1_epi8(va[4]);
							__m128i va5 = _mm_set1_epi8(va[5]);
							__m128i va6 = _mm_set1_epi8(va[6]);
							__m128i va7 = _mm_set1_epi8(va[7]);
							__m128i va8 = _mm_set1_epi8(va[8]);
							__m128i va9 = _mm_set1_epi8(va[9]);
							__m128i va10 = _mm_set1_epi8(va[10]);
							__m128i va11 = _mm_set1_epi8(va[11]);
							__m128i va12 = _mm_set1_epi8(va[12]);
							__m128i va13 = _mm_set1_epi8(va[13]);
							__m128i va14 = _mm_set1_epi8(va[14]);
							__m128i va15 = _mm_set1_epi8(va[15]);
							
							__m128i vb0 = _mm_loadu_si128((__m128i*)vb);

							_mm256_epi16_epi32(vb0, va0, _sum0_0, _sum0_1);
							_mm256_epi16_epi32(vb0, va1, _sum1_0, _sum1_1);
							_mm256_epi16_epi32(vb0, va2, _sum2_0, _sum2_1);
							_mm256_epi16_epi32(vb0, va3, _sum3_0, _sum3_1);
							_mm256_epi16_epi32(vb0, va4, _sum4_0, _sum4_1);
							_mm256_epi16_epi32(vb0, va5, _sum5_0, _sum5_1);
							_mm256_epi16_epi32(vb0, va6, _sum6_0, _sum6_1);
							_mm256_epi16_epi32(vb0, va7, _sum7_0, _sum7_1);
							_mm256_epi16_epi32(vb0, va8, _sum8_0, _sum8_1);
							_mm256_epi16_epi32(vb0, va9, _sum9_0, _sum9_1);
							_mm256_epi16_epi32(vb0, va10, _sum10_0, _sum10_1);
							_mm256_epi16_epi32(vb0, va11, _sum11_0, _sum11_1);
							_mm256_epi16_epi32(vb0, va12, _sum12_0, _sum12_1);
							_mm256_epi16_epi32(vb0, va13, _sum13_0, _sum13_1);
							_mm256_epi16_epi32(vb0, va14, _sum14_0, _sum14_1);
							_mm256_epi16_epi32(vb0, va15, _sum15_0, _sum15_1);
							va += 16;
							vb += 16;
						}
						_mm256_storeu_si256((__m256i*)sum0, _sum0_0);
						_mm256_storeu_si256((__m256i*)(sum0 + 8), _sum0_1);
						_mm256_storeu_si256((__m256i*)sum1, _sum1_0);
						_mm256_storeu_si256((__m256i*)(sum1 + 8), _sum1_1);
						_mm256_storeu_si256((__m256i*)sum2, _sum2_0);
						_mm256_storeu_si256((__m256i*)(sum2 + 8), _sum2_1);
						_mm256_storeu_si256((__m256i*)sum3, _sum1_0);
						_mm256_storeu_si256((__m256i*)(sum3 + 8), _sum3_1);
						_mm256_storeu_si256((__m256i*)sum4, _sum4_0);
						_mm256_storeu_si256((__m256i*)(sum4 + 8), _sum4_1);
						_mm256_storeu_si256((__m256i*)sum5, _sum5_0);
						_mm256_storeu_si256((__m256i*)(sum5 + 8), _sum5_1);
						_mm256_storeu_si256((__m256i*)sum6, _sum6_0);
						_mm256_storeu_si256((__m256i*)(sum6 + 8), _sum6_1);
						_mm256_storeu_si256((__m256i*)sum7, _sum7_0);
						_mm256_storeu_si256((__m256i*)(sum7 + 8), _sum7_1);
						_mm256_storeu_si256((__m256i*)sum8, _sum8_0);
						_mm256_storeu_si256((__m256i*)(sum8 + 8), _sum8_1);
						_mm256_storeu_si256((__m256i*)sum9, _sum9_0);
						_mm256_storeu_si256((__m256i*)(sum9 + 8), _sum9_1);
						_mm256_storeu_si256((__m256i*)sum10, _sum10_0);
						_mm256_storeu_si256((__m256i*)(sum10 + 8), _sum10_1);
						_mm256_storeu_si256((__m256i*)sum11, _sum11_0);
						_mm256_storeu_si256((__m256i*)(sum11 + 8), _sum11_1);
						_mm256_storeu_si256((__m256i*)sum12, _sum12_0);
						_mm256_storeu_si256((__m256i*)(sum12 + 8), _sum12_1);
						_mm256_storeu_si256((__m256i*)sum13, _sum13_0);
						_mm256_storeu_si256((__m256i*)(sum13 + 8), _sum13_1);
						_mm256_storeu_si256((__m256i*)sum14, _sum14_0);
						_mm256_storeu_si256((__m256i*)(sum14 + 8), _sum14_1);
						_mm256_storeu_si256((__m256i*)sum15, _sum15_0);
						_mm256_storeu_si256((__m256i*)(sum15 + 8), _sum15_1);
#else
						

						int k = 0;

						for (; k + 15 < K; k = k + 16)
						{
							for (int n = 0; n < 16; n++)
							{
								sum0[n] += va[0] * vb[n];
								sum1[n] += va[1] * vb[n];
								sum2[n] += va[2] * vb[n];
								sum3[n] += va[3] * vb[n];
								sum4[n] += va[4] * vb[n];
								sum5[n] += va[5] * vb[n];
								sum6[n] += va[6] * vb[n];
								sum7[n] += va[7] * vb[n];
								sum8[n] += va[8] * vb[n];
								sum9[n] += va[9] * vb[n];
								sum10[n] += va[10] * vb[n];
								sum11[n] += va[11] * vb[n];
								sum12[n] += va[12] * vb[n];
								sum13[n] += va[13] * vb[n];
								sum14[n] += va[14] * vb[n];
								sum15[n] += va[15] * vb[n];
								va += 16;

								sum0[n] += va[0] * vb[n + 16];
								sum1[n] += va[1] * vb[n + 16];
								sum2[n] += va[2] * vb[n + 16];
								sum3[n] += va[3] * vb[n + 16];
								sum4[n] += va[4] * vb[n + 16];
								sum5[n] += va[5] * vb[n + 16];
								sum6[n] += va[6] * vb[n + 16];
								sum7[n] += va[7] * vb[n + 16];
								sum8[n] += va[8] * vb[n + 16];
								sum9[n] += va[9] * vb[n + 16];
								sum10[n] += va[10] * vb[n + 16];
								sum11[n] += va[11] * vb[n + 16];
								sum12[n] += va[12] * vb[n + 16];
								sum13[n] += va[13] * vb[n + 16];
								sum14[n] += va[14] * vb[n + 16];
								sum15[n] += va[15] * vb[n + 16];
								va += 16;

								sum0[n] += va[0] * vb[n + 32];
								sum1[n] += va[1] * vb[n + 32];
								sum2[n] += va[2] * vb[n + 32];
								sum3[n] += va[3] * vb[n + 32];
								sum4[n] += va[4] * vb[n + 32];
								sum5[n] += va[5] * vb[n + 32];
								sum6[n] += va[6] * vb[n + 32];
								sum7[n] += va[7] * vb[n + 32];
								sum8[n] += va[8] * vb[n + 32];
								sum9[n] += va[9] * vb[n + 32];
								sum10[n] += va[10] * vb[n + 32];
								sum11[n] += va[11] * vb[n + 32];
								sum12[n] += va[12] * vb[n + 32];
								sum13[n] += va[13] * vb[n + 32];
								sum14[n] += va[14] * vb[n + 32];
								sum15[n] += va[15] * vb[n + 32];
								va += 16;

								sum0[n] += va[0] * vb[n + 48];
								sum1[n] += va[1] * vb[n + 48];
								sum2[n] += va[2] * vb[n + 48];
								sum3[n] += va[3] * vb[n + 48];
								sum4[n] += va[4] * vb[n + 48];
								sum5[n] += va[5] * vb[n + 48];
								sum6[n] += va[6] * vb[n + 48];
								sum7[n] += va[7] * vb[n + 48];
								sum8[n] += va[8] * vb[n + 48];
								sum9[n] += va[9] * vb[n + 48];
								sum10[n] += va[10] * vb[n + 48];
								sum11[n] += va[11] * vb[n + 48];
								sum12[n] += va[12] * vb[n + 48];
								sum13[n] += va[13] * vb[n + 48];
								sum14[n] += va[14] * vb[n + 48];
								sum15[n] += va[15] * vb[n + 48];
								va += 16;

								sum0[n] += va[0] * vb[n + 64];
								sum1[n] += va[1] * vb[n + 64];
								sum2[n] += va[2] * vb[n + 64];
								sum3[n] += va[3] * vb[n + 64];
								sum4[n] += va[4] * vb[n + 64];
								sum5[n] += va[5] * vb[n + 64];
								sum6[n] += va[6] * vb[n + 64];
								sum7[n] += va[7] * vb[n + 64];
								sum8[n] += va[8] * vb[n + 64];
								sum9[n] += va[9] * vb[n + 64];
								sum10[n] += va[10] * vb[n + 64];
								sum11[n] += va[11] * vb[n + 64];
								sum12[n] += va[12] * vb[n + 64];
								sum13[n] += va[13] * vb[n + 64];
								sum14[n] += va[14] * vb[n + 64];
								sum15[n] += va[15] * vb[n + 64];
								va += 16;

								sum0[n] += va[0] * vb[n + 80];
								sum1[n] += va[1] * vb[n + 80];
								sum2[n] += va[2] * vb[n + 80];
								sum3[n] += va[3] * vb[n + 80];
								sum4[n] += va[4] * vb[n + 80];
								sum5[n] += va[5] * vb[n + 80];
								sum6[n] += va[6] * vb[n + 80];
								sum7[n] += va[7] * vb[n + 80];
								sum8[n] += va[8] * vb[n + 80];
								sum9[n] += va[9] * vb[n + 80];
								sum10[n] += va[10] * vb[n + 80];
								sum11[n] += va[11] * vb[n + 80];
								sum12[n] += va[12] * vb[n + 80];
								sum13[n] += va[13] * vb[n + 80];
								sum14[n] += va[14] * vb[n + 80];
								sum15[n] += va[15] * vb[n + 80];
								va += 16;

								sum0[n] += va[0] * vb[n + 96];
								sum1[n] += va[1] * vb[n + 96];
								sum2[n] += va[2] * vb[n + 96];
								sum3[n] += va[3] * vb[n + 96];
								sum4[n] += va[4] * vb[n + 96];
								sum5[n] += va[5] * vb[n + 96];
								sum6[n] += va[6] * vb[n + 96];
								sum7[n] += va[7] * vb[n + 96];
								sum8[n] += va[8] * vb[n + 96];
								sum9[n] += va[9] * vb[n + 96];
								sum10[n] += va[10] * vb[n + 96];
								sum11[n] += va[11] * vb[n + 96];
								sum12[n] += va[12] * vb[n + 96];
								sum13[n] += va[13] * vb[n + 96];
								sum14[n] += va[14] * vb[n + 96];
								sum15[n] += va[15] * vb[n + 96];
								va += 16;

								sum0[n] += va[0] * vb[n + 112];
								sum1[n] += va[1] * vb[n + 112];
								sum2[n] += va[2] * vb[n + 112];
								sum3[n] += va[3] * vb[n + 112];
								sum4[n] += va[4] * vb[n + 112];
								sum5[n] += va[5] * vb[n + 112];
								sum6[n] += va[6] * vb[n + 112];
								sum7[n] += va[7] * vb[n + 112];
								sum8[n] += va[8] * vb[n + 112];
								sum9[n] += va[9] * vb[n + 112];
								sum10[n] += va[10] * vb[n + 112];
								sum11[n] += va[11] * vb[n + 112];
								sum12[n] += va[12] * vb[n + 112];
								sum13[n] += va[13] * vb[n + 112];
								sum14[n] += va[14] * vb[n + 112];
								sum15[n] += va[15] * vb[n + 112];
								va += 16;

								sum0[n] += va[0] * vb[n + 128];
								sum1[n] += va[1] * vb[n + 128];
								sum2[n] += va[2] * vb[n + 128];
								sum3[n] += va[3] * vb[n + 128];
								sum4[n] += va[4] * vb[n + 128];
								sum5[n] += va[5] * vb[n + 128];
								sum6[n] += va[6] * vb[n + 128];
								sum7[n] += va[7] * vb[n + 128];
								sum8[n] += va[8] * vb[n + 128];
								sum9[n] += va[9] * vb[n + 128];
								sum10[n] += va[10] * vb[n + 128];
								sum11[n] += va[11] * vb[n + 128];
								sum12[n] += va[12] * vb[n + 128];
								sum13[n] += va[13] * vb[n + 128];
								sum14[n] += va[14] * vb[n + 128];
								sum15[n] += va[15] * vb[n + 128];
								va += 16;

								sum0[n] += va[0] * vb[n + 144];
								sum1[n] += va[1] * vb[n + 144];
								sum2[n] += va[2] * vb[n + 144];
								sum3[n] += va[3] * vb[n + 144];
								sum4[n] += va[4] * vb[n + 144];
								sum5[n] += va[5] * vb[n + 144];
								sum6[n] += va[6] * vb[n + 144];
								sum7[n] += va[7] * vb[n + 144];
								sum8[n] += va[8] * vb[n + 144];
								sum9[n] += va[9] * vb[n + 144];
								sum10[n] += va[10] * vb[n + 144];
								sum11[n] += va[11] * vb[n + 144];
								sum12[n] += va[12] * vb[n + 144];
								sum13[n] += va[13] * vb[n + 144];
								sum14[n] += va[14] * vb[n + 144];
								sum15[n] += va[15] * vb[n + 144];
								va += 16;

								sum0[n] += va[0] * vb[n + 160];
								sum1[n] += va[1] * vb[n + 160];
								sum2[n] += va[2] * vb[n + 160];
								sum3[n] += va[3] * vb[n + 160];
								sum4[n] += va[4] * vb[n + 160];
								sum5[n] += va[5] * vb[n + 160];
								sum6[n] += va[6] * vb[n + 160];
								sum7[n] += va[7] * vb[n + 160];
								sum8[n] += va[8] * vb[n + 160];
								sum9[n] += va[9] * vb[n + 160];
								sum10[n] += va[10] * vb[n + 160];
								sum11[n] += va[11] * vb[n + 160];
								sum12[n] += va[12] * vb[n + 160];
								sum13[n] += va[13] * vb[n + 160];
								sum14[n] += va[14] * vb[n + 160];
								sum15[n] += va[15] * vb[n + 160];
								va += 16;

								sum0[n] += va[0] * vb[n + 176];
								sum1[n] += va[1] * vb[n + 176];
								sum2[n] += va[2] * vb[n + 176];
								sum3[n] += va[3] * vb[n + 176];
								sum4[n] += va[4] * vb[n + 176];
								sum5[n] += va[5] * vb[n + 176];
								sum6[n] += va[6] * vb[n + 176];
								sum7[n] += va[7] * vb[n + 176];
								sum8[n] += va[8] * vb[n + 176];
								sum9[n] += va[9] * vb[n + 176];
								sum10[n] += va[10] * vb[n + 176];
								sum11[n] += va[11] * vb[n + 176];
								sum12[n] += va[12] * vb[n + 176];
								sum13[n] += va[13] * vb[n + 176];
								sum14[n] += va[14] * vb[n + 176];
								sum15[n] += va[15] * vb[n + 176];
								va += 16;

								sum0[n] += va[0] * vb[n + 192];
								sum1[n] += va[1] * vb[n + 192];
								sum2[n] += va[2] * vb[n + 192];
								sum3[n] += va[3] * vb[n + 192];
								sum4[n] += va[4] * vb[n + 192];
								sum5[n] += va[5] * vb[n + 192];
								sum6[n] += va[6] * vb[n + 192];
								sum7[n] += va[7] * vb[n + 192];
								sum8[n] += va[8] * vb[n + 192];
								sum9[n] += va[9] * vb[n + 192];
								sum10[n] += va[10] * vb[n + 192];
								sum11[n] += va[11] * vb[n + 192];
								sum12[n] += va[12] * vb[n + 192];
								sum13[n] += va[13] * vb[n + 192];
								sum14[n] += va[14] * vb[n + 192];
								sum15[n] += va[15] * vb[n + 192];
								va += 16;

								sum0[n] += va[0] * vb[n + 208];
								sum1[n] += va[1] * vb[n + 208];
								sum2[n] += va[2] * vb[n + 208];
								sum3[n] += va[3] * vb[n + 208];
								sum4[n] += va[4] * vb[n + 208];
								sum5[n] += va[5] * vb[n + 208];
								sum6[n] += va[6] * vb[n + 208];
								sum7[n] += va[7] * vb[n + 208];
								sum8[n] += va[8] * vb[n + 208];
								sum9[n] += va[9] * vb[n + 208];
								sum10[n] += va[10] * vb[n + 208];
								sum11[n] += va[11] * vb[n + 208];
								sum12[n] += va[12] * vb[n + 208];
								sum13[n] += va[13] * vb[n + 208];
								sum14[n] += va[14] * vb[n + 208];
								sum15[n] += va[15] * vb[n + 208];
								va += 16;

								sum0[n] += va[0] * vb[n + 224];
								sum1[n] += va[1] * vb[n + 224];
								sum2[n] += va[2] * vb[n + 224];
								sum3[n] += va[3] * vb[n + 224];
								sum4[n] += va[4] * vb[n + 224];
								sum5[n] += va[5] * vb[n + 224];
								sum6[n] += va[6] * vb[n + 224];
								sum7[n] += va[7] * vb[n + 224];
								sum8[n] += va[8] * vb[n + 224];
								sum9[n] += va[9] * vb[n + 224];
								sum10[n] += va[10] * vb[n + 224];
								sum11[n] += va[11] * vb[n + 224];
								sum12[n] += va[12] * vb[n + 224];
								sum13[n] += va[13] * vb[n + 224];
								sum14[n] += va[14] * vb[n + 224];
								sum15[n] += va[15] * vb[n + 224];
								va += 16;

								sum0[n] += va[0] * vb[n + 240];
								sum1[n] += va[1] * vb[n + 240];
								sum2[n] += va[2] * vb[n + 240];
								sum3[n] += va[3] * vb[n + 240];
								sum4[n] += va[4] * vb[n + 240];
								sum5[n] += va[5] * vb[n + 240];
								sum6[n] += va[6] * vb[n + 240];
								sum7[n] += va[7] * vb[n + 240];
								sum8[n] += va[8] * vb[n + 240];
								sum9[n] += va[9] * vb[n + 240];
								sum10[n] += va[10] * vb[n + 240];
								sum11[n] += va[11] * vb[n + 240];
								sum12[n] += va[12] * vb[n + 240];
								sum13[n] += va[13] * vb[n + 240];
								sum14[n] += va[14] * vb[n + 240];
								sum15[n] += va[15] * vb[n + 240];
								va -= 240;
							}

							va += 256;
							vb += 256;
						}

						for (; k < K; k++)
						{
							for (int n = 0; n < 16; n++)
							{
								sum0[n] += va[0] * vb[n];
								sum1[n] += va[1] * vb[n];
								sum2[n] += va[2] * vb[n];
								sum3[n] += va[3] * vb[n];
								sum4[n] += va[4] * vb[n];
								sum5[n] += va[5] * vb[n];
								sum6[n] += va[6] * vb[n];
								sum7[n] += va[7] * vb[n];
								sum8[n] += va[8] * vb[n];
								sum9[n] += va[9] * vb[n];
								sum10[n] += va[10] * vb[n];
								sum11[n] += va[11] * vb[n];
								sum12[n] += va[12] * vb[n];
								sum13[n] += va[13] * vb[n];
								sum14[n] += va[14] * vb[n];
								sum15[n] += va[15] * vb[n];
							}

							va += 16;
							vb += 16;
						}

						
#endif // __AVX__
						for (int n = 0; n < 16; n++)
						{
							output0[n] = (float)sum0[n] * scale_dequant0 + bias0;
							output1[n] = (float)sum1[n] * scale_dequant0 + bias1;
							output2[n] = (float)sum2[n] * scale_dequant0 + bias2;
							output3[n] = (float)sum3[n] * scale_dequant0 + bias3;
							output4[n] = (float)sum4[n] * scale_dequant0 + bias4;
							output5[n] = (float)sum5[n] * scale_dequant0 + bias5;
							output6[n] = (float)sum6[n] * scale_dequant0 + bias6;
							output7[n] = (float)sum7[n] * scale_dequant0 + bias7;
							output8[n] = (float)sum8[n] * scale_dequant0 + bias8;
							output9[n] = (float)sum9[n] * scale_dequant0 + bias9;
							output10[n] = (float)sum10[n] * scale_dequant0 + bias10;
							output11[n] = (float)sum11[n] * scale_dequant0 + bias11;
							output12[n] = (float)sum12[n] * scale_dequant0 + bias12;
							output13[n] = (float)sum13[n] * scale_dequant0 + bias13;
							output14[n] = (float)sum14[n] * scale_dequant0 + bias14;
							output15[n] = (float)sum15[n] * scale_dequant0 + bias15;
						}
						output0 += 16;
						output1 += 16;
						output2 += 16;
						output3 += 16;
						output4 += 16;
						output5 += 16;
						output6 += 16;
						output7 += 16;
						output8 += 16;
						output9 += 16;
						output10 += 16;
						output11 += 16;
						output12 += 16;
						output13 += 16;
						output14 += 16;
						output15 += 16;
					}

					for (; j < N; j++)
					{
						signed char* vb = bottom_tm_data + (j / 16 + j % 16) * bottom_tm_cstep;;
						signed char* va = kernel_tm_data + (i / 16) * kernel_tm_cstep;


						int sum0 = 0;
						int sum1 = 0;
						int sum2 = 0;
						int sum3 = 0;
						int sum4 = 0;
						int sum5 = 0;
						int sum6 = 0;
						int sum7 = 0;
						int sum8 = 0;
						int sum9 = 0;
						int sum10 = 0;
						int sum11 = 0;
						int sum12 = 0;
						int sum13 = 0;
						int sum14 = 0;
						int sum15 = 0;

						for (int k = 0; k < K; k++)
						{
							sum0 += (int)va[0] * vb[0];
							sum1 += (int)va[1] * vb[0];
							sum2 += (int)va[2] * vb[0];
							sum3 += (int)va[3] * vb[0];
							sum4 += (int)va[4] * vb[0];
							sum5 += (int)va[5] * vb[0];
							sum6 += (int)va[6] * vb[0];
							sum7 += (int)va[7] * vb[0];
							sum8 += (int)va[8] * vb[0];
							sum9 += (int)va[9] * vb[0];
							sum10 += (int)va[10] * vb[0];
							sum11 += (int)va[11] * vb[0];
							sum12 += (int)va[12] * vb[0];
							sum13 += (int)va[13] * vb[0];
							sum14 += (int)va[14] * vb[0];
							sum15 += (int)va[15] * vb[0];
							va += 16;
							vb += 1;
						}

						output0[0] = (float)sum0 * scale_dequant0 + bias0;
						output1[0] = (float)sum1 * scale_dequant0 + bias1;
						output2[0] = (float)sum2 * scale_dequant0 + bias2;
						output3[0] = (float)sum3 * scale_dequant0 + bias3;
						output4[0] = (float)sum4 * scale_dequant0 + bias4;
						output5[0] = (float)sum5 * scale_dequant0 + bias5;
						output6[0] = (float)sum6 * scale_dequant0 + bias6;
						output7[0] = (float)sum7 * scale_dequant0 + bias7;
						output8[0] = (float)sum8 * scale_dequant0 + bias8;
						output9[0] = (float)sum9 * scale_dequant0 + bias9;
						output10[0] = (float)sum10 * scale_dequant0 + bias10;
						output11[0] = (float)sum11 * scale_dequant0 + bias11;
						output12[0] = (float)sum12 * scale_dequant0 + bias12;
						output13[0] = (float)sum13 * scale_dequant0 + bias13;
						output14[0] = (float)sum14 * scale_dequant0 + bias14;
						output15[0] = (float)sum15 * scale_dequant0 + bias15;

						output0++;
						output1++;
						output2++;
						output3++;
						output4++;
						output5++;
						output6++;
						output7++;
						output8++;
						output9++;
						output10++;
						output11++;
						output12++;
						output13++;
						output14++;
						output15++;
					}
				}
			}
		}