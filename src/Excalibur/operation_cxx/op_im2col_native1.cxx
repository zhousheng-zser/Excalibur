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