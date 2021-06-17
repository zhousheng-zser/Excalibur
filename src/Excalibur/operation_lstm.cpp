#include "../../include/Excalibur/operation_lstm.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"
#include "../../include/Excalibur/math_functions.hpp"
#include <random>

#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION)
#include "../../include/Excalibur/avx_mathfun.h"
#endif

namespace glasssix
{
    namespace excalibur
    {
#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION)
        static inline __m256 HorizontalSums(__m256 &v0, __m256 &v1, __m256 &v2, __m256 &v3, __m256 &v4, __m256 &v5, __m256 &v6, __m256 &v7)
        {
            const __m256 s01 = _mm256_hadd_ps(v0, v1);
            const __m256 s23 = _mm256_hadd_ps(v2, v3);
            const __m256 s45 = _mm256_hadd_ps(v4, v5);
            const __m256 s67 = _mm256_hadd_ps(v6, v7);
            const __m256 s0123 = _mm256_hadd_ps(s01, s23);
            const __m256 s4556 = _mm256_hadd_ps(s45, s67);

            // inter-lane shuffle
            const __m256 vb0 = _mm256_blend_ps(s0123, s4556, 0xF0);
            const __m256 vb1 = _mm256_permute2f128_ps(s0123, s4556, 0x21);

            return _mm256_add_ps(vb0, vb1);
        }

        static inline __m128 HorizontalSums(__m256 &v0, __m256 &v1, __m256 &v2, __m256 &v3)
        {
            const __m256 s01 = _mm256_hadd_ps(v0, v1);
            const __m256 s23 = _mm256_hadd_ps(v2, v3);
            const __m256 s0123 = _mm256_hadd_ps(s01, s23);

            return _mm_add_ps(_mm256_extractf128_ps(s0123, 1),
                              _mm256_castps256_ps128(s0123));
        }

        static inline __m128 HorizontalSums(__m256 &v0, __m256 &v1, __m256 &v2)
        {
            const __m256 v3 = _mm256_set1_ps(0.0f);
            const __m256 s01 = _mm256_hadd_ps(v0, v1);
            const __m256 s23 = _mm256_hadd_ps(v2, v3);
            const __m256 s0123 = _mm256_hadd_ps(s01, s23);

            return _mm_add_ps(_mm256_extractf128_ps(s0123, 1),
                              _mm256_castps256_ps128(s0123));
        }

        static inline __m256 sigmoid_avx(__m256 inputs)
        {
            const __m256 one = _mm256_set1_ps(1.0f);
            return _mm256_div_ps(one, _mm256_add_ps(one, exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), inputs))));
        }

        static inline __m256 tanh_avx(__m256 inputs)
        {
            const __m256 one = _mm256_set1_ps(1.0f);
            const __m256 two = _mm256_set1_ps(2.0f);
            return _mm256_fmsub_ps(sigmoid_avx(_mm256_mul_ps(inputs, two)), two, one);
        }
#endif
        template <typename Dtype>
        operation_lstm<Dtype>::operation_lstm(const operation_param &param) : operation<Dtype>(param)
        {
            auto attrs = split_string(param.specific_params_, " ");
            for (size_t i = 0; i < attrs.size(); i++)
            {
                if (split_string(attrs[i], "=")[0] == "0")
                {
                    num_output_ = atoi(split_string(attrs[i], "=")[1].c_str());
                    hidden_.reset(new memory::tensor<float>(num_output_, this->params_.device_, memory::NCHW));
                    cell_.reset(new memory::tensor<float>(num_output_, this->params_.device_, memory::NCHW));
#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION)
                    gates_.reset(new memory::tensor<float>(4, num_output_, this->params_.device_, memory::NCHW));
#else

                    gates_.reset(new memory::tensor<float>(num_output_, 4, this->params_.device_, memory::NCHW));
#endif
                    std::memset(hidden_->mutable_cpu_data(), 0, num_output_ * sizeof(float));
                    std::memset(cell_->mutable_cpu_data(), 0, num_output_ * sizeof(float));
                    std::memset(gates_->mutable_cpu_data(), 0, num_output_ * 4 * sizeof(float));
                }
                else if (split_string(attrs[i], "=")[0] == "1")
                {
                    weight_data_size_ = atoi(split_string(attrs[i], "=")[1].c_str());
                }
                else if (split_string(attrs[i], "=")[0] == "2")
                {
                    direction_ = atoi(split_string(attrs[i], "=")[1].c_str());
                }
                else if (split_string(attrs[i], "=")[0] == "-23330")
                {
                    //do nothing
                }
                else
                {
                    LOG(FATAL) << "Un-supported InnerProduct Attribution " << split_string(attrs[i], "=")[0];
                }
            }
        }

        template <typename Dtype>
        int operation_lstm<Dtype>::init_weights(FILE *fp)
        {
            int quantize_tag;
            fread(&quantize_tag, 1, sizeof(int), fp);
            int mem = 0;
            if (quantize_tag == 0)
            {
                int num_directions = direction_ == 2 ? 2 : 1;
                this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(weight_data_size_, this->params_.device_, memory::NCHW, nullptr)));
                fread(this->weights_f32_[0]->mutable_cpu_data(), 1, weight_data_size_ * sizeof(float), fp);
                mem += weight_data_size_ * sizeof(float);
                fread(&quantize_tag, 1, sizeof(int), fp);
                this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(4 * num_output_ * num_directions, this->params_.device_, memory::NCHW, nullptr)));
                fread(this->weights_f32_[1]->mutable_cpu_data(), 1, 4 * num_output_ * num_directions * sizeof(float), fp);
                mem += 4 * num_output_ * num_directions * sizeof(float);
                fread(&quantize_tag, 1, sizeof(int), fp);
                this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(4 * num_output_ * num_output_ * num_directions, this->params_.device_, memory::NCHW, nullptr)));
                fread(this->weights_f32_[2]->mutable_cpu_data(), 1, 4 * num_output_ * num_output_ * num_directions * sizeof(float), fp);
                mem += 4 * num_output_ * num_output_ * num_directions * sizeof(float);
                return mem;
            }
            else
            {
                NOT_IMPLEMENTED;
                return 0;
            }
        }

        template <typename Dtype>
        int operation_lstm<Dtype>::init_weights()
        {
            std::default_random_engine e;
            std::normal_distribution<float> n(0, 0.3);
            std::uniform_int_distribution<int> u(-128, 127);
            int mem = 0;
            if (!this->params_.int8_quantization_)
            {
                int num_directions = direction_ == 2 ? 2 : 1;
                this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(weight_data_size_, this->params_.device_, memory::NCHW, nullptr)));
                for (size_t i = 0; i < weight_data_size_; i++)
                {
                    this->weights_f32_[0]->mutable_cpu_data()[i] = n(e);
                }
                mem += weight_data_size_ * sizeof(float);
                this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(4 * num_output_ * num_directions, this->params_.device_, memory::NCHW, nullptr)));
                for (size_t i = 0; i < 4 * num_output_ * num_directions; i++)
                {
                    this->weights_f32_[1]->mutable_cpu_data()[i] = n(e);
                }
                mem += 4 * num_output_ * num_directions * sizeof(float);
                this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(4 * num_output_ * num_output_ * num_directions, this->params_.device_, memory::NCHW, nullptr)));
                for (size_t i = 0; i < 4 * num_output_ * num_output_ * num_directions; i++)
                {
                    this->weights_f32_[2]->mutable_cpu_data()[i] = n(e);
                }
                mem += 4 * num_output_ * num_output_ * num_directions * sizeof(float);
            }
            else
            {
                NOT_IMPLEMENTED;
            }
            return mem;
        }

        template <typename Dtype>
        void operation_lstm<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms,
                                                    std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);
            int num = bottoms[0]->num();

            int T = bottoms[0]->height();
            int num_directions = direction_ == 2 ? 2 : 1;

            if (bottoms[0]->order() == memory::NCHW)
            {
                tops[0].reset(new memory::tensor<float>(std::vector<int>{num, 1, T, num_directions * num_output_}, this->params_.device_, memory::NCHW));
                if (direction_ == 0 || direction_ == 1)
                {
                    lstm_cpu_f32(bottoms[0], tops[0], direction_);
                }
                else if (direction_ == 2)
                {
                    auto top_forward = std::make_shared<memory::tensor<float>>(num, T, num_output_, this->params_.device_);
                    auto top_reverse = std::make_shared<memory::tensor<float>>(num, T, num_output_, this->params_.device_);
                    lstm_cpu_f32(bottoms[0], top_forward, 0);
                    std::memset(hidden_->mutable_cpu_data(), 0, num_output_ * sizeof(float));
                    std::memset(cell_->mutable_cpu_data(), 0, num_output_ * sizeof(float));
                    std::memset(gates_->mutable_cpu_data(), 0, num_output_ * 4 * sizeof(float));
                    lstm_cpu_f32(bottoms[0], top_reverse, 1);

                    const float *top_forward_data = top_forward->cpu_data();
                    const float *top_reverse_data = top_reverse->cpu_data();
                    for (int n = 0; n < num; n++)
                    {
                        for (int i = 0; i < T; i++)
                        {
                            const float *pf = top_forward_data + n * T * num_output_ + i * num_output_;
                            const float *pr = top_reverse_data + n * T * num_output_ + i * num_output_;
                            float *ptr = tops[0]->mutable_cpu_data() + n * T * num_output_ * num_directions + i * num_output_ * num_directions;

                            std::copy(pf, pf + num_output_, ptr);
                            std::copy(pr, pr + num_output_, ptr + num_output_);
                        }
                    }
                }
            }
            else
            {
                NOT_IMPLEMENTED;
            }
        }

        template <typename Dtype>
        void operation_lstm<Dtype>::lstm_cpu_f32(const std::shared_ptr<memory::tensor<float>> &bottom, std::shared_ptr<memory::tensor<float>> &top, int reverse)
        {
            int num = bottom->num();
            int size = bottom->width();
            int T = bottom->height();

            int w_xc = this->weight_data_size_ / (direction_ == 2 ? 2 : 1) / num_output_ / 4;
            const float *weight_xc_data = this->weights_f32_[0]->cpu_data() + reverse * w_xc * num_output_ * 4;
            const float *bias_c_data = this->weights_f32_[1]->cpu_data() + reverse * 4 * num_output_;
            const float *bias_c_I = bias_c_data + 0 * num_output_;
            const float *bias_c_F = bias_c_data + 1 * num_output_;
            const float *bias_c_O = bias_c_data + 2 * num_output_;
            const float *bias_c_G = bias_c_data + 3 * num_output_;
            const float *weight_hc_data = this->weights_f32_[2]->cpu_data() + reverse * num_output_ * num_output_ * 4;

            float *hidden_data = hidden_->mutable_cpu_data();
            float *cell_data = cell_->mutable_cpu_data();
            float *gates = gates_->mutable_cpu_data();

            for (int n = 0; n < num; n++)
            {
                const float *bottom_data = bottom->cpu_data() + n * size * T;
                float *top_data = top->mutable_cpu_data() + n * T * num_output_;

                // unroll
                for (int t = 0; t < T; t++)
                {
                    // clip hidden by continuation indicator
                    // h_cont_{t-1} = cont_t * h_{t-1}
                    // h_cont_{t-1} = h_{t-1} if cont_t == 1
                    //                0       otherwise
                    // calculate hidden
                    // gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
                    int ti = reverse ? T - 1 - t : t;

#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION)
                    int remain_output = (num_output_ >> 1) << 1;
                    for (int q = 0; q + 1 < num_output_; q += 2)
                    {
                        const float *x = bottom_data + ti * size;
                        const float *hidden_ptr_r = hidden_data;
                        float *gates_data_I = gates;
                        float *gates_data_F = gates + num_output_;
                        float *gates_data_O = gates + num_output_ * 2;
                        float *gates_data_G = gates + num_output_ * 3;

                        // gate I F O G
                        const float *weight_xc_I_0 = weight_xc_data + w_xc * (num_output_ * 0 + q);
                        const float *weight_xc_F_0 = weight_xc_data + w_xc * (num_output_ * 1 + q);
                        const float *weight_xc_O_0 = weight_xc_data + w_xc * (num_output_ * 2 + q);
                        const float *weight_xc_G_0 = weight_xc_data + w_xc * (num_output_ * 3 + q);
                        const float *weight_xc_I_1 = weight_xc_data + w_xc * (num_output_ * 0 + (q + 1));
                        const float *weight_xc_F_1 = weight_xc_data + w_xc * (num_output_ * 1 + (q + 1));
                        const float *weight_xc_O_1 = weight_xc_data + w_xc * (num_output_ * 2 + (q + 1));
                        const float *weight_xc_G_1 = weight_xc_data + w_xc * (num_output_ * 3 + (q + 1));

                        const float *weight_hc_I_0 = weight_hc_data + num_output_ * (num_output_ * 0 + q);
                        const float *weight_hc_F_0 = weight_hc_data + num_output_ * (num_output_ * 1 + q);
                        const float *weight_hc_O_0 = weight_hc_data + num_output_ * (num_output_ * 2 + q);
                        const float *weight_hc_G_0 = weight_hc_data + num_output_ * (num_output_ * 3 + q);
                        const float *weight_hc_I_1 = weight_hc_data + num_output_ * (num_output_ * 0 + (q + 1));
                        const float *weight_hc_F_1 = weight_hc_data + num_output_ * (num_output_ * 1 + (q + 1));
                        const float *weight_hc_O_1 = weight_hc_data + num_output_ * (num_output_ * 2 + (q + 1));
                        const float *weight_hc_G_1 = weight_hc_data + num_output_ * (num_output_ * 3 + (q + 1));

                        // float I = bias_c_I[q];
                        // float F = bias_c_F[q];
                        // float O = bias_c_O[q];
                        // float G = bias_c_G[q];
                        __m256 _sumI_0 = _mm256_setzero_ps();
                        __m256 _sumF_0 = _mm256_setzero_ps();
                        __m256 _sumO_0 = _mm256_setzero_ps();
                        __m256 _sumG_0 = _mm256_setzero_ps();
                        __m256 _sumI_1 = _mm256_setzero_ps();
                        __m256 _sumF_1 = _mm256_setzero_ps();
                        __m256 _sumO_1 = _mm256_setzero_ps();
                        __m256 _sumG_1 = _mm256_setzero_ps();
                        int nn_num_size = size >> 3;
                        int remain_size = size & 7;
                        for (; nn_num_size > 0; nn_num_size--)
                        {
                            __m256 xi = _mm256_loadu_ps(x);
                            _sumI_0 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_I_0), xi, _sumI_0);
                            _sumF_0 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_F_0), xi, _sumF_0);
                            _sumO_0 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_O_0), xi, _sumO_0);
                            _sumG_0 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_G_0), xi, _sumG_0);
                            _sumI_1 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_I_1), xi, _sumI_1);
                            _sumF_1 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_F_1), xi, _sumF_1);
                            _sumO_1 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_O_1), xi, _sumO_1);
                            _sumG_1 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_G_1), xi, _sumG_1);
                            x += 8;
                            weight_xc_I_0 += 8;
                            weight_xc_F_0 += 8;
                            weight_xc_O_0 += 8;
                            weight_xc_G_0 += 8;
                            weight_xc_I_1 += 8;
                            weight_xc_F_1 += 8;
                            weight_xc_O_1 += 8;
                            weight_xc_G_1 += 8;
                        }
                        int nn_num_output = num_output_ >> 3;
                        int remain_num_output = num_output_ & 7;
                        for (; nn_num_output > 0; nn_num_output--)
                        {
                            __m256 h_cont = _mm256_loadu_ps(hidden_ptr_r);

                            _sumI_0 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_I_0), h_cont, _sumI_0);
                            _sumF_0 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_F_0), h_cont, _sumF_0);
                            _sumO_0 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_O_0), h_cont, _sumO_0);
                            _sumG_0 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_G_0), h_cont, _sumG_0);
                            _sumI_1 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_I_1), h_cont, _sumI_1);
                            _sumF_1 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_F_1), h_cont, _sumF_1);
                            _sumO_1 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_O_1), h_cont, _sumO_1);
                            _sumG_1 = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_G_1), h_cont, _sumG_1);
                            hidden_ptr_r += 8;
                            weight_hc_I_0 += 8;
                            weight_hc_F_0 += 8;
                            weight_hc_O_0 += 8;
                            weight_hc_G_0 += 8;
                            weight_hc_I_1 += 8;
                            weight_hc_F_1 += 8;
                            weight_hc_O_1 += 8;
                            weight_hc_G_1 += 8;
                        }
                        float sums[8];
                        _mm256_storeu_ps(sums, HorizontalSums(_sumI_0, _sumF_0, _sumO_0, _sumG_0, _sumI_1, _sumF_1, _sumO_1, _sumG_1));
                        sums[0] += bias_c_I[q];
                        sums[1] += bias_c_F[q];
                        sums[2] += bias_c_O[q];
                        sums[3] += bias_c_G[q];
                        sums[4] += bias_c_I[q + 1];
                        sums[5] += bias_c_F[q + 1];
                        sums[6] += bias_c_O[q + 1];
                        sums[7] += bias_c_G[q + 1];

                        for (; remain_size > 0; remain_size--)
                        {
                            float xi = *x;
                            sums[0] += *weight_xc_I_0 * xi;
                            sums[1] += *weight_xc_F_0 * xi;
                            sums[2] += *weight_xc_O_0 * xi;
                            sums[3] += *weight_xc_G_0 * xi;
                            sums[4] += *weight_xc_I_1 * xi;
                            sums[5] += *weight_xc_F_1 * xi;
                            sums[6] += *weight_xc_O_1 * xi;
                            sums[7] += *weight_xc_G_1 * xi;
                            x++;
                            weight_xc_I_0++;
                            weight_xc_F_0++;
                            weight_xc_O_0++;
                            weight_xc_G_0++;
                            weight_xc_I_1++;
                            weight_xc_F_1++;
                            weight_xc_O_1++;
                            weight_xc_G_1++;
                        }

                        for (; remain_num_output > 0; remain_num_output--)
                        {
                            float h_cont = *hidden_ptr_r;
                            sums[0] += *weight_hc_I_0 * h_cont;
                            sums[1] += *weight_hc_F_0 * h_cont;
                            sums[2] += *weight_hc_O_0 * h_cont;
                            sums[3] += *weight_hc_G_0 * h_cont;
                            sums[4] += *weight_hc_I_1 * h_cont;
                            sums[5] += *weight_hc_F_1 * h_cont;
                            sums[6] += *weight_hc_O_1 * h_cont;
                            sums[7] += *weight_hc_G_1 * h_cont;
                            hidden_ptr_r++;
                            weight_hc_I_0++;
                            weight_hc_F_0++;
                            weight_hc_O_0++;
                            weight_hc_G_0++;
                            weight_hc_I_1++;
                            weight_hc_F_1++;
                            weight_hc_O_1++;
                            weight_hc_G_1++;
                        }
                        gates_data_I[q] = sums[0];
                        gates_data_F[q] = sums[1];
                        gates_data_O[q] = sums[2];
                        gates_data_G[q] = sums[3];
                        gates_data_I[q + 1] = sums[4];
                        gates_data_F[q + 1] = sums[5];
                        gates_data_O[q + 1] = sums[6];
                        gates_data_G[q + 1] = sums[7];
                    }

                    for (int q = remain_output; q < num_output_; q++)
                    {
                        const float *x = bottom_data + ti * size;
                        const float *hidden_ptr_r = hidden_data;
                        float *gates_data_I = gates;
                        float *gates_data_F = gates + num_output_;
                        float *gates_data_O = gates + num_output_ * 2;
                        float *gates_data_G = gates + num_output_ * 3;
                        // gate I F O G
                        const float *weight_xc_I = weight_xc_data + w_xc * (num_output_ * 0 + q);
                        const float *weight_xc_F = weight_xc_data + w_xc * (num_output_ * 1 + q);
                        const float *weight_xc_O = weight_xc_data + w_xc * (num_output_ * 2 + q);
                        const float *weight_xc_G = weight_xc_data + w_xc * (num_output_ * 3 + q);
                        const float *weight_hc_I = weight_hc_data + num_output_ * (num_output_ * 0 + q);
                        const float *weight_hc_F = weight_hc_data + num_output_ * (num_output_ * 1 + q);
                        const float *weight_hc_O = weight_hc_data + num_output_ * (num_output_ * 2 + q);
                        const float *weight_hc_G = weight_hc_data + num_output_ * (num_output_ * 3 + q);

                        // float I = bias_c_I[q];
                        // float F = bias_c_F[q];
                        // float O = bias_c_O[q];
                        // float G = bias_c_G[q];
                        __m256 _sumI = _mm256_setzero_ps();
                        __m256 _sumF = _mm256_setzero_ps();
                        __m256 _sumO = _mm256_setzero_ps();
                        __m256 _sumG = _mm256_setzero_ps();
                        int nn_num_size = size >> 3;
                        int remain_size = size & 7;
                        for (; nn_num_size > 0; nn_num_size--)
                        {
                            __m256 xi = _mm256_loadu_ps(x);
                            _sumI = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_I), xi, _sumI);
                            _sumF = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_F), xi, _sumF);
                            _sumO = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_O), xi, _sumO);
                            _sumG = _mm256_fmadd_ps(_mm256_loadu_ps(weight_xc_G), xi, _sumG);
                            x += 8;
                            weight_xc_I += 8;
                            weight_xc_F += 8;
                            weight_xc_O += 8;
                            weight_xc_G += 8;
                        }
                        int nn_num_output = num_output_ >> 3;
                        int remain_num_output = num_output_ & 7;
                        for (; nn_num_output > 0; nn_num_output--)
                        {
                            __m256 h_cont = _mm256_loadu_ps(hidden_ptr_r);

                            _sumI = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_I), h_cont, _sumI);
                            _sumF = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_F), h_cont, _sumF);
                            _sumO = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_O), h_cont, _sumO);
                            _sumG = _mm256_fmadd_ps(_mm256_loadu_ps(weight_hc_G), h_cont, _sumG);
                            hidden_ptr_r += 8;
                            weight_hc_I += 8;
                            weight_hc_F += 8;
                            weight_hc_O += 8;
                            weight_hc_G += 8;
                        }
                        float sums[4];
                        _mm_storeu_ps(sums, HorizontalSums(_sumI, _sumF, _sumO, _sumG));
                        sums[0] += bias_c_I[q];
                        sums[1] += bias_c_F[q];
                        sums[2] += bias_c_O[q];
                        sums[3] += bias_c_G[q];

                        for (; remain_size > 0; remain_size--)
                        {
                            float xi = *x;
                            sums[0] += *weight_xc_I * xi;
                            sums[1] += *weight_xc_F * xi;
                            sums[2] += *weight_xc_O * xi;
                            sums[3] += *weight_xc_G * xi;
                            x++;
                            weight_xc_I++;
                            weight_xc_F++;
                            weight_xc_O++;
                            weight_xc_G++;
                        }

                        for (; remain_num_output > 0; remain_num_output--)
                        {
                            float h_cont = *hidden_ptr_r;
                            sums[0] += *weight_hc_I * h_cont;
                            sums[1] += *weight_hc_F * h_cont;
                            sums[2] += *weight_hc_O * h_cont;
                            sums[3] += *weight_hc_G * h_cont;
                            hidden_ptr_r++;
                            weight_hc_I++;
                            weight_hc_F++;
                            weight_hc_O++;
                            weight_hc_G++;
                        }
                        gates_data_I[q] = sums[0];
                        gates_data_F[q] = sums[1];
                        gates_data_O[q] = sums[2];
                        gates_data_G[q] = sums[3];
                    }
                    // lstm unit
                    // sigmoid(I)
                    // sigmoid(F)
                    // sigmoid(O)
                    // tanh(G)
                    // c_t := f_t .* c_{t-1} + i_t .* g_t
                    // h_t := o_t .* tanh[c_t]
                    float *output_data = top_data + ti * num_output_;
                    float *cell_ptr = cell_data;
                    float *hidden_ptr = hidden_data;

                    const float *gates_data_I = gates;
                    const float *gates_data_F = gates + num_output_;
                    const float *gates_data_O = gates + num_output_ * 2;
                    const float *gates_data_G = gates + num_output_ * 3;

                    int nn_activation = num_output_ >> 3;
                    int remain_activations = num_output_ & 7;
                    for (; nn_activation > 0; nn_activation--)
                    {
                        __m256 I = sigmoid_avx(_mm256_loadu_ps(gates_data_I));
                        __m256 F = sigmoid_avx(_mm256_loadu_ps(gates_data_F));
                        __m256 O = sigmoid_avx(_mm256_loadu_ps(gates_data_O));
                        __m256 G = tanh_avx(_mm256_loadu_ps(gates_data_G));
                        __m256 cell2 = _mm256_add_ps(_mm256_mul_ps(F, _mm256_loadu_ps(cell_ptr)), _mm256_mul_ps(I, G));
                        __m256 H = _mm256_mul_ps(O, tanh_avx(cell2));
                        _mm256_storeu_ps(cell_ptr, cell2);
                        _mm256_storeu_ps(hidden_ptr, H);
                        _mm256_storeu_ps(output_data, H);
                        cell_ptr += 8;
                        output_data += 8;
                        hidden_ptr += 8;
                        gates_data_I += 8;
                        gates_data_F += 8;
                        gates_data_O += 8;
                        gates_data_G += 8;
                    }

                    for (; remain_activations > 0; remain_activations--)
                    {
                        float I = *gates_data_I;
                        float F = *gates_data_F;
                        float O = *gates_data_O;
                        float G = *gates_data_G;

                        I = 1.f / (1.f + exp(-I));
                        F = 1.f / (1.f + exp(-F));
                        O = 1.f / (1.f + exp(-O));
                        G = tanh(G);
                        float cell2 = F * *cell_ptr + I * G;
                        float H = O * tanh(cell2);
                        *cell_ptr = cell2;
                        *hidden_ptr = H;
                        *output_data = H;
                        cell_ptr++;
                        output_data++;
                        hidden_ptr++;
                        gates_data_I++;
                        gates_data_F++;
                        gates_data_O++;
                        gates_data_G++;
                    }
#else
                    for (int q = 0; q < num_output_; q++)
                    {
                        const float *x = bottom_data + ti * size;
                        float *gates_data = gates + q * 4;

                        // gate I F O G
                        const float *weight_xc_I = weight_xc_data + (num_output_ * 0 + q) * w_xc;
                        const float *weight_xc_F = weight_xc_data + (num_output_ * 1 + q) * w_xc;
                        const float *weight_xc_O = weight_xc_data + (num_output_ * 2 + q) * w_xc;
                        const float *weight_xc_G = weight_xc_data + (num_output_ * 3 + q) * w_xc;

                        const float *weight_hc_I = weight_hc_data + (num_output_ * 0 + q) * num_output_;
                        const float *weight_hc_F = weight_hc_data + (num_output_ * 1 + q) * num_output_;
                        const float *weight_hc_O = weight_hc_data + (num_output_ * 2 + q) * num_output_;
                        const float *weight_hc_G = weight_hc_data + (num_output_ * 3 + q) * num_output_;

                        float I = bias_c_I[q]; // -0.0290763229
                        float F = bias_c_F[q]; // -0.244906723
                        float O = bias_c_O[q]; // 1.97673798
                        float G = bias_c_G[q]; // -0.977587759

                        for (int i = 0; i < size; i++)
                        {
                            float xi = x[i];

                            I += weight_xc_I[i] * xi;
                            F += weight_xc_F[i] * xi;
                            O += weight_xc_O[i] * xi;
                            G += weight_xc_G[i] * xi;
                        }

                        for (int i = 0; i < num_output_; i++)
                        {
                            float h_cont = hidden_data[i];

                            I += weight_hc_I[i] * h_cont;
                            F += weight_hc_F[i] * h_cont;
                            O += weight_hc_O[i] * h_cont;
                            G += weight_hc_G[i] * h_cont;
                        }

                        gates_data[0] = I;
                        gates_data[1] = F;
                        gates_data[2] = O;
                        gates_data[3] = G;
                    }

                    // lstm unit
                    // sigmoid(I)
                    // sigmoid(F)
                    // sigmoid(O)
                    // tanh(G)
                    // c_t := f_t .* c_{t-1} + i_t .* g_t
                    // h_t := o_t .* tanh[c_t]
                    float *output_data = top_data + ti * num_output_;
                    for (int q = 0; q < num_output_; q++)
                    {
                        const float *gates_data = gates + q * 4;

                        float I = gates_data[0];
                        float F = gates_data[1];
                        float O = gates_data[2];
                        float G = gates_data[3];

                        I = 1.f / (1.f + exp(-I));
                        F = 1.f / (1.f + exp(-F));
                        O = 1.f / (1.f + exp(-O));
                        G = tanh(G);

                        float cell2 = F * cell_data[q] + I * G;
                        float H = O * tanh(cell2);
                        cell_data[q] = cell2;
                        hidden_data[q] = H;
                        output_data[q] = H;
                    }
#endif
                }
            }
        }

        template <typename Dtype>
        void operation_lstm<Dtype>::forward_gpu_f32(
#ifdef USE_CUDA
            cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
            cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
            const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms,
            std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            NOT_IMPLEMENTED;
        }

        //#ifndef USE_CUDA
        //		STUB_GPU(operation_lstm);
        //#endif
        INSTANCE_CLASS(operation_lstm);
        REGISTE(operation_lstm);
    }
}