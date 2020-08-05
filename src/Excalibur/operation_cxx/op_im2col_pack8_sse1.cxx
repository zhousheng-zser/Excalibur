for (; k + 3 < L; k = k + 4)
							{
								// k0
								__m256 _va0 = _mm256_broadcast_ss(va);
								__m256 _va1 = _mm256_broadcast_ss(va + 1);
								__m256 _va2 = _mm256_broadcast_ss(va + 2);
								__m256 _va3 = _mm256_broadcast_ss(va + 3);
								__m256 _vb0 = _mm256_loadu_ps(vb);
								__m256 _vb1 = _mm256_loadu_ps(vb + 8);
								__m256 _vb2 = _mm256_loadu_ps(vb + 16);
								__m256 _vb3 = _mm256_loadu_ps(vb + 24);
								_sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0);    // sum0 = (a00-a07) * k00
								_sum1 = _mm256_fmadd_ps(_vb0, _va1, _sum1);    // sum1 = (a00-a07) * k10
								_sum2 = _mm256_fmadd_ps(_vb0, _va2, _sum2);    // sum2 = (a00-a07) * k20
								_sum3 = _mm256_fmadd_ps(_vb0, _va3, _sum3);    // sum3 = (a00-a07) * k30
								_va0 = _mm256_broadcast_ss(va + 4);
								_va1 = _mm256_broadcast_ss(va + 5);
								_va2 = _mm256_broadcast_ss(va + 6);
								_va3 = _mm256_broadcast_ss(va + 7);
								_sum4 = _mm256_fmadd_ps(_vb0, _va0, _sum4);    // sum4 = (a00-a07) * k40
								_sum5 = _mm256_fmadd_ps(_vb0, _va1, _sum5);    // sum5 = (a00-a07) * k50
								_sum6 = _mm256_fmadd_ps(_vb0, _va2, _sum6);    // sum6 = (a00-a07) * k60
								_sum7 = _mm256_fmadd_ps(_vb0, _va3, _sum7);    // sum7 = (a00-a07) * k70

								va += 8;

								// k1
								_va0 = _mm256_broadcast_ss(va);
								_va1 = _mm256_broadcast_ss(va + 1);
								_va2 = _mm256_broadcast_ss(va + 2);
								_va3 = _mm256_broadcast_ss(va + 3);
								_sum0 = _mm256_fmadd_ps(_vb1, _va0, _sum0);    // sum0 += (a10-a17) * k01
								_sum1 = _mm256_fmadd_ps(_vb1, _va1, _sum1);    // sum1 += (a10-a17) * k11
								_sum2 = _mm256_fmadd_ps(_vb1, _va2, _sum2);    // sum2 += (a10-a17) * k21
								_sum3 = _mm256_fmadd_ps(_vb1, _va3, _sum3);    // sum3 += (a10-a17) * k31
								_va0 = _mm256_broadcast_ss(va + 4);
								_va1 = _mm256_broadcast_ss(va + 5);
								_va2 = _mm256_broadcast_ss(va + 6);
								_va3 = _mm256_broadcast_ss(va + 7);
								_sum4 = _mm256_fmadd_ps(_vb1, _va0, _sum4);    // sum4 += (a10-a17) * k41
								_sum5 = _mm256_fmadd_ps(_vb1, _va1, _sum5);    // sum5 += (a10-a17) * k51
								_sum6 = _mm256_fmadd_ps(_vb1, _va2, _sum6);    // sum6 += (a10-a17) * k61
								_sum7 = _mm256_fmadd_ps(_vb1, _va3, _sum7);    // sum7 += (a10-a17) * k71

								va += 8;

								// k2
								_va0 = _mm256_broadcast_ss(va);
								_va1 = _mm256_broadcast_ss(va + 1);
								_va2 = _mm256_broadcast_ss(va + 2);
								_va3 = _mm256_broadcast_ss(va + 3);
								_sum0 = _mm256_fmadd_ps(_vb2, _va0, _sum0);    // sum0 += (a20-a27) * k02
								_sum1 = _mm256_fmadd_ps(_vb2, _va1, _sum1);    // sum1 += (a20-a27) * k12
								_sum2 = _mm256_fmadd_ps(_vb2, _va2, _sum2);    // sum2 += (a20-a27) * k22
								_sum3 = _mm256_fmadd_ps(_vb2, _va3, _sum3);    // sum3 += (a20-a27) * k32
								_va0 = _mm256_broadcast_ss(va + 4);
								_va1 = _mm256_broadcast_ss(va + 5);
								_va2 = _mm256_broadcast_ss(va + 6);
								_va3 = _mm256_broadcast_ss(va + 7);
								_sum4 = _mm256_fmadd_ps(_vb2, _va0, _sum4);    // sum4 += (a20-a27) * k42
								_sum5 = _mm256_fmadd_ps(_vb2, _va1, _sum5);    // sum5 += (a20-a27) * k52
								_sum6 = _mm256_fmadd_ps(_vb2, _va2, _sum6);    // sum6 += (a20-a27) * k62
								_sum7 = _mm256_fmadd_ps(_vb2, _va3, _sum7);    // sum7 += (a20-a27) * k72  

								va += 8;

								// k3
								_va0 = _mm256_broadcast_ss(va);
								_va1 = _mm256_broadcast_ss(va + 1);
								_va2 = _mm256_broadcast_ss(va + 2);
								_va3 = _mm256_broadcast_ss(va + 3);
								_sum0 = _mm256_fmadd_ps(_vb3, _va0, _sum0);    // sum0 += (a30-a37) * k03
								_sum1 = _mm256_fmadd_ps(_vb3, _va1, _sum1);    // sum1 += (a30-a37) * k13
								_sum2 = _mm256_fmadd_ps(_vb3, _va2, _sum2);    // sum2 += (a30-a37) * k23
								_sum3 = _mm256_fmadd_ps(_vb3, _va3, _sum3);    // sum3 += (a30-a37) * k33
								_va0 = _mm256_broadcast_ss(va + 4);
								_va1 = _mm256_broadcast_ss(va + 5);
								_va2 = _mm256_broadcast_ss(va + 6);
								_va3 = _mm256_broadcast_ss(va + 7);
								_sum4 = _mm256_fmadd_ps(_vb3, _va0, _sum4);    // sum4 += (a30-a37) * k43
								_sum5 = _mm256_fmadd_ps(_vb3, _va1, _sum5);    // sum5 += (a30-a37) * k53
								_sum6 = _mm256_fmadd_ps(_vb3, _va2, _sum6);    // sum6 += (a30-a37) * k63
								_sum7 = _mm256_fmadd_ps(_vb3, _va3, _sum7);    // sum7 += (a30-a37) * k73                      

								va += 8;
								vb += 32;
							}