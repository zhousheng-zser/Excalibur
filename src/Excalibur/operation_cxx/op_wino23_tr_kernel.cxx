									for (int n = 0; n < 4; n++)
									{
										d0[n] = r0[n];
										d1[n] = r1[n];
										d2[n] = r2[n];
										d3[n] = r3[n];
									}
									// w = B_t * d
									for (int n = 0; n < 4; n++)
									{
										w0[n] = d0[n] - d2[n];
										w1[n] = d1[n] + d2[n];
										w2[n] = d2[n] - d1[n];
										w3[n] = d3[n] - d1[n];
									}
									// transpose d to d_t
									{
										t0[0] = w0[0]; t1[0] = w0[1]; t2[0] = w0[2]; t3[0] = w0[3];
										t0[1] = w1[0]; t1[1] = w1[1]; t2[1] = w1[2]; t3[1] = w1[3];
										t0[2] = w2[0]; t1[2] = w2[1]; t2[2] = w2[2]; t3[2] = w2[3];
										t0[3] = w3[0]; t1[3] = w3[1]; t2[3] = w3[2]; t3[3] = w3[3];
									}
									// d = B_t * d_t
									for (int n = 0; n < 4; n++)
									{
										d0[n] = t0[n] - t2[n];
										d1[n] = t1[n] + t2[n];
										d2[n] = t2[n] - t1[n];
										d3[n] = t3[n] - t1[n];
									}
									// save to out_tm
									for (int n = 0; n < 4; n++)
									{
										out_tm0[n] = d0[n];
										out_tm0[n + 4] = d1[n];
										out_tm0[n + 8] = d2[n];
										out_tm0[n + 12] = d3[n];
									}