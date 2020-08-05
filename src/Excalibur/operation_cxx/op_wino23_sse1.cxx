							for (; q + 3 < inch; q += 4)
								{
									const float* r0 = bottom_blob_tm_data + q * tiles * 16 + i * 16;
									const float* r1 = bottom_blob_tm_data + (q + 1) * tiles * 16 + i * 16;
									const float* r2 = bottom_blob_tm_data + (q + 2) * tiles * 16 + i * 16;
									const float* r3 = bottom_blob_tm_data + (q + 3) * tiles * 16 + i * 16;
									const float* k0 = kernel0_tm + q * 16;
									const float* k1 = kernel1_tm + q * 16;
									const float* k2 = kernel2_tm + q * 16;
									const float* k3 = kernel3_tm + q * 16;
									//r
									__m256 _r0 = _mm256_loadu_ps(r0);
									__m256 _r0n = _mm256_loadu_ps(r0 + 8);
									// k0
									__m256 _k0 = _mm256_loadu_ps(k0); __m256 _k0n = _mm256_loadu_ps(k0 + 8);
									__m256 _k1 = _mm256_loadu_ps(k1); __m256 _k1n = _mm256_loadu_ps(k1 + 8);
									__m256 _k2 = _mm256_loadu_ps(k2); __m256 _k2n = _mm256_loadu_ps(k2 + 8);
									__m256 _k3 = _mm256_loadu_ps(k3); __m256 _k3n = _mm256_loadu_ps(k3 + 8);

									_sum0 = _mm256_fmadd_ps(_r0, _k0, _sum0); _sum0n = _mm256_fmadd_ps(_r0n, _k0n, _sum0n);
									_sum1 = _mm256_fmadd_ps(_r0, _k1, _sum1); _sum1n = _mm256_fmadd_ps(_r0n, _k1n, _sum1n);
									_sum2 = _mm256_fmadd_ps(_r0, _k2, _sum2); _sum2n = _mm256_fmadd_ps(_r0n, _k2n, _sum2n);
									_sum3 = _mm256_fmadd_ps(_r0, _k3, _sum3); _sum3n = _mm256_fmadd_ps(_r0n, _k3n, _sum3n);

									// k1
									_r0 = _mm256_loadu_ps(r1); _r0n = _mm256_loadu_ps(r1 + 8);
									_k0 = _mm256_loadu_ps(k0 + 16); _k0n = _mm256_loadu_ps(k0 + 24);
									_k1 = _mm256_loadu_ps(k1 + 16); _k1n = _mm256_loadu_ps(k1 + 24);
									_k2 = _mm256_loadu_ps(k2 + 16); _k2n = _mm256_loadu_ps(k2 + 24);
									_k3 = _mm256_loadu_ps(k3 + 16); _k3n = _mm256_loadu_ps(k3 + 24);
									_sum0 = _mm256_fmadd_ps(_r0, _k0, _sum0); _sum0n = _mm256_fmadd_ps(_r0n, _k0n, _sum0n);
									_sum1 = _mm256_fmadd_ps(_r0, _k1, _sum1); _sum1n = _mm256_fmadd_ps(_r0n, _k1n, _sum1n);
									_sum2 = _mm256_fmadd_ps(_r0, _k2, _sum2); _sum2n = _mm256_fmadd_ps(_r0n, _k2n, _sum2n);
									_sum3 = _mm256_fmadd_ps(_r0, _k3, _sum3); _sum3n = _mm256_fmadd_ps(_r0n, _k3n, _sum3n);
									// k2   
									_r0 = _mm256_loadu_ps(r2); _r0n = _mm256_loadu_ps(r2 + 8);
									_k0 = _mm256_loadu_ps(k0 + 32); _k0n = _mm256_loadu_ps(k0 + 40);
									_k1 = _mm256_loadu_ps(k1 + 32); _k1n = _mm256_loadu_ps(k1 + 40);
									_k2 = _mm256_loadu_ps(k2 + 32); _k2n = _mm256_loadu_ps(k2 + 40);
									_k3 = _mm256_loadu_ps(k3 + 32); _k3n = _mm256_loadu_ps(k3 + 40);

									_sum0 = _mm256_fmadd_ps(_r0, _k0, _sum0); _sum0n = _mm256_fmadd_ps(_r0n, _k0n, _sum0n);
									_sum1 = _mm256_fmadd_ps(_r0, _k1, _sum1); _sum1n = _mm256_fmadd_ps(_r0n, _k1n, _sum1n);
									_sum2 = _mm256_fmadd_ps(_r0, _k2, _sum2); _sum2n = _mm256_fmadd_ps(_r0n, _k2n, _sum2n);
									_sum3 = _mm256_fmadd_ps(_r0, _k3, _sum3); _sum3n = _mm256_fmadd_ps(_r0n, _k3n, _sum3n);
									// k3   
									_r0 = _mm256_loadu_ps(r3); _r0n = _mm256_loadu_ps(r3 + 8);
									_k0 = _mm256_loadu_ps(k0 + 48); _k0n = _mm256_loadu_ps(k0 + 56);
									_k1 = _mm256_loadu_ps(k1 + 48); _k1n = _mm256_loadu_ps(k1 + 56);
									_k2 = _mm256_loadu_ps(k2 + 48); _k2n = _mm256_loadu_ps(k2 + 56);
									_k3 = _mm256_loadu_ps(k3 + 48); _k3n = _mm256_loadu_ps(k3 + 56);
									_sum0 = _mm256_fmadd_ps(_r0, _k0, _sum0); _sum0n = _mm256_fmadd_ps(_r0n, _k0n, _sum0n);
									_sum1 = _mm256_fmadd_ps(_r0, _k1, _sum1); _sum1n = _mm256_fmadd_ps(_r0n, _k1n, _sum1n);
									_sum2 = _mm256_fmadd_ps(_r0, _k2, _sum2); _sum2n = _mm256_fmadd_ps(_r0n, _k2n, _sum2n);
									_sum3 = _mm256_fmadd_ps(_r0, _k3, _sum3); _sum3n = _mm256_fmadd_ps(_r0n, _k3n, _sum3n);
								}