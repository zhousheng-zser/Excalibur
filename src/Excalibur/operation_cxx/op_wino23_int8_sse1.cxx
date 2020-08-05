							for (; q + 3 < inch; q += 4)
								{
									const short* r0 = bottom_blob_tm_data + q * tiles * 16 + i * 16;
									const short* r1 = bottom_blob_tm_data + (q + 1) * tiles * 16 + i * 16;
									const short* r2 = bottom_blob_tm_data + (q + 2) * tiles * 16 + i * 16;
									const short* r3 = bottom_blob_tm_data + (q + 3) * tiles * 16 + i * 16;
									const short* k0 = kernel0_tm + q * 16;
									const short* k1 = kernel1_tm + q * 16;
									const short* k2 = kernel2_tm + q * 16;
									const short* k3 = kernel3_tm + q * 16;

									__m128i _r0 = _mm_loadu_si128((__m128i*)r0); __m128i _r0n = _mm_loadu_si128((__m128i*)(r0 + 8));
									__m128i _k0 = _mm_loadu_si128((__m128i*)k0); __m128i _k0n = _mm_loadu_si128((__m128i*)(k0 + 8));
									__m128i _k1 = _mm_loadu_si128((__m128i*)k1); __m128i _k1n = _mm_loadu_si128((__m128i*)(k1 + 8));
									__m128i _k2 = _mm_loadu_si128((__m128i*)k2); __m128i _k2n = _mm_loadu_si128((__m128i*)(k2 + 8));
									__m128i _k3 = _mm_loadu_si128((__m128i*)k3); __m128i _k3n = _mm_loadu_si128((__m128i*)(k3 + 8));

									_sum0_0 = _mm256__epi32(_r0, _k0, _sum0_0); _sum0_1 = _mm256__epi32(_r0n, _k0n, _sum0_1);
									_sum1_0 = _mm256__epi32(_r0, _k1, _sum1_0); _sum1_1 = _mm256__epi32(_r0n, _k1n, _sum1_1);
									_sum2_0 = _mm256__epi32(_r0, _k2, _sum2_0); _sum2_1 = _mm256__epi32(_r0n, _k2n, _sum2_1);
									_sum3_0 = _mm256__epi32(_r0, _k3, _sum3_0); _sum3_1 = _mm256__epi32(_r0n, _k3n, _sum3_1);


									_r0 = _mm_loadu_si128((__m128i*)r1); _r0n = _mm_loadu_si128((__m128i*)(r1 + 8));
									_k0 = _mm_loadu_si128((__m128i*)(k0 + 16)); _k0n = _mm_loadu_si128((__m128i*)(k0 + 24));
									_k1 = _mm_loadu_si128((__m128i*)(k1 + 16)); _k1n = _mm_loadu_si128((__m128i*)(k1 + 24));
									_k2 = _mm_loadu_si128((__m128i*)(k2 + 16)); _k2n = _mm_loadu_si128((__m128i*)(k2 + 24));
									_k3 = _mm_loadu_si128((__m128i*)(k3 + 16)); _k3n = _mm_loadu_si128((__m128i*)(k3 + 24));
									_sum0_0 = _mm256__epi32(_r0, _k0, _sum0_0); _sum0_1 = _mm256__epi32(_r0n, _k0n, _sum0_1);
									_sum1_0 = _mm256__epi32(_r0, _k1, _sum1_0); _sum1_1 = _mm256__epi32(_r0n, _k1n, _sum1_1);
									_sum2_0 = _mm256__epi32(_r0, _k2, _sum2_0); _sum2_1 = _mm256__epi32(_r0n, _k2n, _sum2_1);
									_sum3_0 = _mm256__epi32(_r0, _k3, _sum3_0); _sum3_1 = _mm256__epi32(_r0n, _k3n, _sum3_1);

									_r0 = _mm_loadu_si128((__m128i*)r2); _r0n = _mm_loadu_si128((__m128i*)(r2 + 8));
									_k0 = _mm_loadu_si128((__m128i*)(k0 + 32)); _k0n = _mm_loadu_si128((__m128i*)(k0 + 40));
									_k1 = _mm_loadu_si128((__m128i*)(k1 + 32)); _k1n = _mm_loadu_si128((__m128i*)(k1 + 40));
									_k2 = _mm_loadu_si128((__m128i*)(k2 + 32)); _k2n = _mm_loadu_si128((__m128i*)(k2 + 40));
									_k3 = _mm_loadu_si128((__m128i*)(k3 + 32)); _k3n = _mm_loadu_si128((__m128i*)(k3 + 40));

									_sum0_0 = _mm256__epi32(_r0, _k0, _sum0_0); _sum0_1 = _mm256__epi32(_r0n, _k0n, _sum0_1);
									_sum1_0 = _mm256__epi32(_r0, _k1, _sum1_0); _sum1_1 = _mm256__epi32(_r0n, _k1n, _sum1_1);
									_sum2_0 = _mm256__epi32(_r0, _k2, _sum2_0); _sum2_1 = _mm256__epi32(_r0n, _k2n, _sum2_1);
									_sum3_0 = _mm256__epi32(_r0, _k3, _sum3_0); _sum3_1 = _mm256__epi32(_r0n, _k3n, _sum3_1);


									_r0 = _mm_loadu_si128((__m128i*)r3); _r0n = _mm_loadu_si128((__m128i*)(r3 + 8));
									_k0 = _mm_loadu_si128((__m128i*)(k0 + 48)); _k0n = _mm_loadu_si128((__m128i*)(k0 + 56));
									_k1 = _mm_loadu_si128((__m128i*)(k1 + 48)); _k1n = _mm_loadu_si128((__m128i*)(k1 + 56));
									_k2 = _mm_loadu_si128((__m128i*)(k2 + 48)); _k2n = _mm_loadu_si128((__m128i*)(k2 + 56));
									_k3 = _mm_loadu_si128((__m128i*)(k3 + 48)); _k3n = _mm_loadu_si128((__m128i*)(k3 + 56));

									_sum0_0 = _mm256__epi32(_r0, _k0, _sum0_0); _sum0_1 = _mm256__epi32(_r0n, _k0n, _sum0_1);
									_sum1_0 = _mm256__epi32(_r0, _k1, _sum1_0); _sum1_1 = _mm256__epi32(_r0n, _k1n, _sum1_1);
									_sum2_0 = _mm256__epi32(_r0, _k2, _sum2_0); _sum2_1 = _mm256__epi32(_r0n, _k2n, _sum2_1);
									_sum3_0 = _mm256__epi32(_r0, _k3, _sum3_0); _sum3_1 = _mm256__epi32(_r0n, _k3n, _sum3_1);

								}