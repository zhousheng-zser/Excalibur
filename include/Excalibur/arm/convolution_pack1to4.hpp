static void convolution_transform_kernel_pack1to4_neon(const std::shared_ptr<glasssix::memory::tensor<float>>& weight_data, std::shared_ptr<glasssix::memory::tensor<float>>& weight_data_pack1to4, int num_input, int num_output, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = 4b-kw-kh-inch-outch/4b

    weight_data_pack1to4.reset(new glasssix::memory::tensor<float>(num_output / 4, num_input, maxk * 4, -1));

    const float* weight_data_ptr = weight_data->cpu_data();
    const int weight_cstep = maxk * num_input;
    float* weight_data_pack1to4_ptr = weight_data_pack1to4->mutable_cpu_data();
    const int weight_pack4_cstep = weight_cstep * 4;
    for (int q = 0; q + 3 < num_output; q += 4)
    {
        const float* k0 = weight_data_ptr + (q) * weight_cstep;
        const float* k1 = weight_data_ptr + (q + 1) * weight_cstep;
        const float* k2 = weight_data_ptr + (q + 2) * weight_cstep;
        const float* k3 = weight_data_ptr + (q + 3) * weight_cstep;

        float* g0 = weight_data_pack1to4_ptr + (q / 4) * weight_pack4_cstep;

        for (int p = 0; p < num_input; p++)
        {
            const float* k00 = k0 + (p) * maxk;
            const float* k10 = k1 + (p) * maxk;
            const float* k20 = k2 + (p) * maxk;
            const float* k30 = k3 + (p) * maxk;

            float* g00 = g0 + (p) * maxk * 4;

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];

                g00 += 4;
            }
        }
    }
}

static void convolution_pack1to4_neon(const std::shared_ptr<glasssix::memory::tensor<float>>& bottom, std::shared_ptr<glasssix::memory::tensor<float>>& top, const std::shared_ptr<glasssix::memory::tensor<float>>& weight_data_pack1to4, const std::shared_ptr<glasssix::memory::tensor<float>>& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    int w = bottom->width();
    int channels = bottom->channels();

    int outw = top->width() / 4;
    int outh = top->height();
    int outch = top->channels();

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    const float* bias_data_ptr = bias_data.get() ? bias_data->cpu_data() : nullptr;
    const float* bottom_data = bottom->cpu_data();
    const int bottom_cstep = bottom->count(2, 4);
    float* top_data = top->mutable_cpu_data();
    const int top_cstep = top->count(2, 4);
    const float* weight_data_pack1to4_ptr = weight_data_pack1to4->cpu_data();
    // num_output
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
    for (int p = 0; p < outch; p++)
    {
        float* outptr = top_data + (p) * top_cstep;

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float32x4_t _sum = vdupq_n_f32(0.f);

                if (bias_data_ptr)
                {
                    _sum = vld1q_f32(bias_data_ptr + p * 4);
                }

                const float* kptr = weight_data_pack1to4_ptr + maxk * channels * p * 4;

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const float* m = bottom_data + (q) * bottom_cstep;
                    const float* sptr = m + (i * stride_h) * w + j * stride_w;

                    for (int k = 0; k < maxk; k++) // 29.23
                    {
                        float32x4_t _val = vdupq_n_f32(sptr[space_ofs[k]]);
                        float32x4_t _w = vld1q_f32(kptr);
                        _sum = vmlaq_f32(_sum, _val, _w);

                        kptr += 4;
                    }
                }

                //_sum = activation_ps(_sum, activation_type, activation_params);

                vst1q_f32(outptr + j * 4, _sum);
            }

            outptr += outw * 4;
        }
    }
}
