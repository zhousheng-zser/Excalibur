						for (; k + 7 < L; k = k + 8)
							{
									//std::cout << (int)va[0] << " ";
									__m128i _va0 = _mm_set1_epi8(va[0]);
									__m128i _va1 = _mm_set1_epi8(va[1]);
									__m128i _va2 = _mm_set1_epi8(va[2]);
									__m128i _va3 = _mm_set1_epi8(va[3]);
									
									__m128i _vb0 = _mm_loadl_epi64((__m128i*)vb);
									__m128i _vb1 = _mm_loadl_epi64((__m128i*)(vb + 8));
									__m128i _vb2 = _mm_loadl_epi64((__m128i*)(vb + 16));
									__m128i _vb3 = _mm_loadl_epi64((__m128i*)(vb + 24));
									__m128i _vb4 = _mm_loadl_epi64((__m128i*)(vb + 32));
									__m128i _vb5 = _mm_loadl_epi64((__m128i*)(vb + 40));
									__m128i _vb6 = _mm_loadl_epi64((__m128i*)(vb + 48));
									__m128i _vb7 = _mm_loadl_epi64((__m128i*)(vb + 56));

									_sum0 = _mm256_add_epi16_epi32(_vb0, _va0, _sum0);
									_sum1 = _mm256_add_epi16_epi32(_vb0, _va1, _sum1);
									_sum2 = _mm256_add_epi16_epi32(_vb0, _va2, _sum2);
									_sum3 = _mm256_add_epi16_epi32(_vb0, _va3, _sum3);

									 _va0 = _mm_set1_epi8(va[4]);
									 _va1 = _mm_set1_epi8(va[5]);
									 _va2 = _mm_set1_epi8(va[6]);
									 _va3 = _mm_set1_epi8(va[7]);

									_sum4 = _mm256_add_epi16_epi32(_vb0, _va0, _sum4);
									_sum5 = _mm256_add_epi16_epi32(_vb0, _va1, _sum5);
									_sum6 = _mm256_add_epi16_epi32(_vb0, _va2, _sum6);
									_sum7 = _mm256_add_epi16_epi32(_vb0, _va3, _sum7);

									va += 8;


									_va0 = _mm_set1_epi8(va[0]);
									_va1 = _mm_set1_epi8(va[1]);
									_va2 = _mm_set1_epi8(va[2]);
									_va3 = _mm_set1_epi8(va[3]);

									_sum0 = _mm256_add_epi16_epi32(_vb1, _va0, _sum0);
									_sum1 = _mm256_add_epi16_epi32(_vb1, _va1, _sum1);
									_sum2 = _mm256_add_epi16_epi32(_vb1, _va2, _sum2);
									_sum3 = _mm256_add_epi16_epi32(_vb1, _va3, _sum3);

									_va0 = _mm_set1_epi8(va[4]);
									_va1 = _mm_set1_epi8(va[5]);
									_va2 = _mm_set1_epi8(va[6]);
									_va3 = _mm_set1_epi8(va[7]);

									_sum4 = _mm256_add_epi16_epi32(_vb1, _va0, _sum4);
									_sum5 = _mm256_add_epi16_epi32(_vb1, _va1, _sum5);
									_sum6 = _mm256_add_epi16_epi32(_vb1, _va2, _sum6);
									_sum7 = _mm256_add_epi16_epi32(_vb1, _va3, _sum7);

									va += 8;

									_va0 = _mm_set1_epi8(va[0]);
									_va1 = _mm_set1_epi8(va[1]);
									_va2 = _mm_set1_epi8(va[2]);
									_va3 = _mm_set1_epi8(va[3]);

									_sum0 = _mm256_add_epi16_epi32(_vb2, _va0, _sum0);
									_sum1 = _mm256_add_epi16_epi32(_vb2, _va1, _sum1);
									_sum2 = _mm256_add_epi16_epi32(_vb2, _va2, _sum2);
									_sum3 = _mm256_add_epi16_epi32(_vb2, _va3, _sum3);

									_va0 = _mm_set1_epi8(va[4]);
									_va1 = _mm_set1_epi8(va[5]);
									_va2 = _mm_set1_epi8(va[6]);
									_va3 = _mm_set1_epi8(va[7]);

									_sum4 = _mm256_add_epi16_epi32(_vb2, _va0, _sum4);
									_sum5 = _mm256_add_epi16_epi32(_vb2, _va1, _sum5);
									_sum6 = _mm256_add_epi16_epi32(_vb2, _va2, _sum6);
									_sum7 = _mm256_add_epi16_epi32(_vb2, _va3, _sum7);

									va += 8;

									_va0 = _mm_set1_epi8(va[0]);
									_va1 = _mm_set1_epi8(va[1]);
									_va2 = _mm_set1_epi8(va[2]);
									_va3 = _mm_set1_epi8(va[3]);

									_sum0 = _mm256_add_epi16_epi32(_vb3, _va0, _sum0);
									_sum1 = _mm256_add_epi16_epi32(_vb3, _va1, _sum1);
									_sum2 = _mm256_add_epi16_epi32(_vb3, _va2, _sum2);
									_sum3 = _mm256_add_epi16_epi32(_vb3, _va3, _sum3);

									_va0 = _mm_set1_epi8(va[4]);
									_va1 = _mm_set1_epi8(va[5]);
									_va2 = _mm_set1_epi8(va[6]);
									_va3 = _mm_set1_epi8(va[7]);

									_sum4 = _mm256_add_epi16_epi32(_vb3, _va0, _sum4);
									_sum5 = _mm256_add_epi16_epi32(_vb3, _va1, _sum5);
									_sum6 = _mm256_add_epi16_epi32(_vb3, _va2, _sum6);
									_sum7 = _mm256_add_epi16_epi32(_vb3, _va3, _sum7);


									va += 8;

									_va0 = _mm_set1_epi8(va[0]);
									_va1 = _mm_set1_epi8(va[1]);
									_va2 = _mm_set1_epi8(va[2]);
									_va3 = _mm_set1_epi8(va[3]);

									_sum0 = _mm256_add_epi16_epi32(_vb4, _va0, _sum0);
									_sum1 = _mm256_add_epi16_epi32(_vb4, _va1, _sum1);
									_sum2 = _mm256_add_epi16_epi32(_vb4, _va2, _sum2);
									_sum3 = _mm256_add_epi16_epi32(_vb4, _va3, _sum3);

									_va0 = _mm_set1_epi8(va[4]);
									_va1 = _mm_set1_epi8(va[5]);
									_va2 = _mm_set1_epi8(va[6]);
									_va3 = _mm_set1_epi8(va[7]);

									_sum4 = _mm256_add_epi16_epi32(_vb4, _va0, _sum4);
									_sum5 = _mm256_add_epi16_epi32(_vb4, _va1, _sum5);
									_sum6 = _mm256_add_epi16_epi32(_vb4, _va2, _sum6);
									_sum7 = _mm256_add_epi16_epi32(_vb4, _va3, _sum7);


									va += 8;

									_va0 = _mm_set1_epi8(va[0]);
									_va1 = _mm_set1_epi8(va[1]);
									_va2 = _mm_set1_epi8(va[2]);
									_va3 = _mm_set1_epi8(va[3]);

									_sum0 = _mm256_add_epi16_epi32(_vb5, _va0, _sum0);
									_sum1 = _mm256_add_epi16_epi32(_vb5, _va1, _sum1);
									_sum2 = _mm256_add_epi16_epi32(_vb5, _va2, _sum2);
									_sum3 = _mm256_add_epi16_epi32(_vb5, _va3, _sum3);

									_va0 = _mm_set1_epi8(va[4]);
									_va1 = _mm_set1_epi8(va[5]);
									_va2 = _mm_set1_epi8(va[6]);
									_va3 = _mm_set1_epi8(va[7]);

									_sum4 = _mm256_add_epi16_epi32(_vb5, _va0, _sum4);
									_sum5 = _mm256_add_epi16_epi32(_vb5, _va1, _sum5);
									_sum6 = _mm256_add_epi16_epi32(_vb5, _va2, _sum6);
									_sum7 = _mm256_add_epi16_epi32(_vb5, _va3, _sum7);


									va += 8;

									_va0 = _mm_set1_epi8(va[0]);
									_va1 = _mm_set1_epi8(va[1]);
									_va2 = _mm_set1_epi8(va[2]);
									_va3 = _mm_set1_epi8(va[3]);

									_sum0 = _mm256_add_epi16_epi32(_vb6, _va0, _sum0);
									_sum1 = _mm256_add_epi16_epi32(_vb6, _va1, _sum1);
									_sum2 = _mm256_add_epi16_epi32(_vb6, _va2, _sum2);
									_sum3 = _mm256_add_epi16_epi32(_vb6, _va3, _sum3);

									_va0 = _mm_set1_epi8(va[4]);
									_va1 = _mm_set1_epi8(va[5]);
									_va2 = _mm_set1_epi8(va[6]);
									_va3 = _mm_set1_epi8(va[7]);

									_sum4 = _mm256_add_epi16_epi32(_vb6, _va0, _sum4);
									_sum5 = _mm256_add_epi16_epi32(_vb6, _va1, _sum5);
									_sum6 = _mm256_add_epi16_epi32(_vb6, _va2, _sum6);
									_sum7 = _mm256_add_epi16_epi32(_vb6, _va3, _sum7);


									va += 8;

									_va0 = _mm_set1_epi8(va[0]);
									_va1 = _mm_set1_epi8(va[1]);
									_va2 = _mm_set1_epi8(va[2]);
									_va3 = _mm_set1_epi8(va[3]);

									_sum0 = _mm256_add_epi16_epi32(_vb7, _va0, _sum0);
									_sum1 = _mm256_add_epi16_epi32(_vb7, _va1, _sum1);
									_sum2 = _mm256_add_epi16_epi32(_vb7, _va2, _sum2);
									_sum3 = _mm256_add_epi16_epi32(_vb7, _va3, _sum3);

									_va0 = _mm_set1_epi8(va[4]);
									_va1 = _mm_set1_epi8(va[5]);
									_va2 = _mm_set1_epi8(va[6]);
									_va3 = _mm_set1_epi8(va[7]);

									_sum4 = _mm256_add_epi16_epi32(_vb7, _va0, _sum4);
									_sum5 = _mm256_add_epi16_epi32(_vb7, _va1, _sum5);
									_sum6 = _mm256_add_epi16_epi32(_vb7, _va2, _sum6);
									_sum7 = _mm256_add_epi16_epi32(_vb7, _va3, _sum7);


									va += 8;
									vb += 64;
							}