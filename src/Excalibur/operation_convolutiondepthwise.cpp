#include "../../include/Excalibur/operation_reflector.hpp"
#include "../../include/Excalibur/operation_convolutiondepthwise.hpp"
#include "./operation_make_border.hpp"
#include <algorithm>
#include "../../include/Primitives/profiler.hpp"
using namespace std;

namespace glasssix
{
    namespace excalibur
    {
        template <typename Dtype>
        operation_convolutiondepthwise<Dtype>::operation_convolutiondepthwise(const operation_param &param) : operation_convolution<Dtype>(param)
        {
        }

        template <typename Dtype>
        void operation_convolutiondepthwise<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms,
                                                                    std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);
            CHECK_EQ(this->output_channel_, this->group_);

            this->num_ = bottoms[0]->num();
            this->input_dim_h_ = bottoms[0]->height();
            this->input_dim_w_ = bottoms[0]->width();
            this->input_channel_ = bottoms[0]->channels();

            CHECK_EQ(this->input_channel_, this->output_channel_);
            this->output_dim_h_ = (this->input_dim_h_ + this->pad_bottom_ + this->pad_top_ - this->kernel_size_h_) / this->stride_h_ + 1;
            this->output_dim_w_ = (this->input_dim_w_ + this->pad_left_ + this->pad_right_ - this->kernel_size_w_) / this->stride_w_ + 1;

            tops[0].reset(new memory::tensor<float>(std::vector<int>{this->num_, this->output_channel_, this->output_dim_h_, this->output_dim_w_}, this->params_.device_, memory::NCHW, nullptr));

            const float *weights_data = this->weights_f32_[0]->cpu_data();

            if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 1 && this->stride_w_ == 1))
            {
                forward_k3s1_f32(bottoms[0], tops[0]);
            }
            else if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 2 && this->stride_w_ == 2))
            {
                forward_k3s2_f32(bottoms[0], tops[0]);
            }
            else if (this->input_dim_h_ == 4 && this->input_dim_w_ == 4 && this->kernel_size_h_ == 4 && this->kernel_size_w_ == 4)
            {
                tops[0].reset(new memory::tensor<float>(std::vector<int>{this->num_, this->output_channel_, this->output_dim_h_, this->output_dim_w_},
                                                        bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));

                for (int n = 0; n < tops[0]->num(); n++)
                {
                    const float *weights_data = this->weights_f32_[0]->cpu_data();
                    const float *bias_data = nullptr;
                    if (this->bias_term_)
                    {
                        bias_data = this->weights_f32_[1]->cpu_data();
                    }

                    const float *bottom_data = bottoms[0]->cpu_data() + n * bottoms[0]->count(1, 4);
                    float *top_data = tops[0]->mutable_cpu_data() + n * tops[0]->count(1, 4);

                    for (int i = 0; i < tops[0]->count(1, 4); i++)
                    {
#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX //AVX
                        float zero_val = 0.f;
                        __m256 _sum1 = _mm256_broadcast_ss(&zero_val);
                        __m256 _r1 = _mm256_loadu_ps(bottom_data);
                        __m256 _k1 = _mm256_loadu_ps(weights_data);
                        _sum1 = _mm256_fmadd_ps(_r1, _k1, _sum1);

                        __m256 _r2 = _mm256_loadu_ps(bottom_data + 8);
                        __m256 _k2 = _mm256_loadu_ps(weights_data + 8);
                        _sum1 = _mm256_fmadd_ps(_r2, _k2, _sum1);
                        *top_data = _mm256_sum_ps(_sum1);
                        *top_data += *bias_data;
                        bias_data += 1;
                        top_data += 1;
                        bottom_data += 16;
                        weights_data += 16;
#endif
#if __ARM_NEON
                        float32x4_t _sum1 = vdupq_n_f32(0);
                        float32x4_t _sum2 = vdupq_n_f32(0);

                        float32x4_t _r1 = vld1q_f32(bottom_data);
                        float32x4_t _r2 = vld1q_f32(bottom_data + 4);
                        float32x4_t _k1 = vld1q_f32(weights_data);
                        float32x4_t _k2 = vld1q_f32(weights_data + 4);

                        _sum1 = vmlaq_f32(_sum1, _r1, _k1);
                        _sum2 = vmlaq_f32(_sum2, _r2, _k2);

                        float32x4_t _r3 = vld1q_f32(bottom_data + 8);
                        float32x4_t _r4 = vld1q_f32(bottom_data + 12);
                        float32x4_t _k3 = vld1q_f32(weights_data + 8);
                        float32x4_t _k4 = vld1q_f32(weights_data + 12);

                        _sum1 = vmlaq_f32(_sum1, _r3, _k3);
                        _sum2 = vmlaq_f32(_sum2, _r4, _k4);

                        _sum1 = vaddq_f32(_sum1, _sum2);

                        *top_data = vgetq_lane_f32(_sum1, 0) + vgetq_lane_f32(_sum1, 1) + vgetq_lane_f32(_sum1, 2) + vgetq_lane_f32(_sum1, 3);
                        *top_data += *bias_data;
                        bias_data += 1;
                        top_data += 1;
                        bottom_data += 16;
                        weights_data += 16;
#endif
                    }
                }
            }
            else
            {
                operation_convolution<Dtype>::forward_cpu_f32(bottoms, tops);
                // const float *bottom_data = bottoms[0]->cpu_data();
                // const float *weights_data = this->weights_f32_[0]->cpu_data();
                // const float *bias_data = nullptr;
                // float *top_data = nullptr;
                // if (this->bias_term_)
                // {
                //     bias_data = this->weights_f32_[1]->cpu_data();
                // }
                // switch (bottoms[0]->order())
                // {
                // case memory::NCHW:
                //     tops[0].reset(new memory::tensor<float>(std::vector<int>{this->num_, this->output_channel_, this->output_dim_h_, this->output_dim_w_},
                //                                             bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
                //     for (size_t n = 0; n < this->num_; n++)
                //     {
                //         top_data = tops[0]->mutable_cpu_data() + n * tops[0]->count(1, 4);
                //         for (int i = 0; i < tops[0]->count(1, 4); i++)
                //         {
                //             const int pw = i % this->output_dim_w_;
                //             const int ph = (i / this->output_dim_w_) % this->output_dim_h_;
                //             const int c = (i / this->output_dim_w_ / this->output_dim_h_) % this->output_channel_;
                //             const int n_step = i / this->output_dim_w_ / this->output_dim_h_ / this->output_channel_;
                //             int hstart = ph * this->stride_h_ - this->pad_top_;
                //             int wstart = pw * this->stride_w_ - this->pad_left_;
                //             int hend = std::min(hstart + this->kernel_size_h_, this->input_dim_h_ + this->pad_bottom_);
                //             int wend = std::min(wstart + this->kernel_size_w_, this->input_dim_w_ + this->pad_right_);
                //             hstart = std::max(hstart, 0);
                //             wstart = std::max(wstart, 0);
                //             hend = std::min(hend, this->input_dim_h_);
                //             wend = std::min(wend, this->input_dim_w_);
                //             float aveval = 0;
                //             const float *bottom_slice =
                //                 bottom_data + n * bottoms[0]->count(1, 4) + (n_step * this->output_channel_ + c) * this->input_dim_h_ * this->input_dim_w_;
                //             const float *weight_slice =
                //                 weights_data + c * this->kernel_size_h_ * this->kernel_size_w_;
                //             int khstart = hend < this->kernel_size_h_ ? this->kernel_size_h_ - hend : 0;
                //             int kwstart = wend < this->kernel_size_w_ ? this->kernel_size_w_ - wend : 0;
                //             for (int h = hstart; h < hend; ++h)
                //             {
                //                 for (int w = wstart; w < wend; ++w)
                //                 {
                //                     aveval += bottom_slice[h * this->input_dim_h_ + w] * weight_slice[(khstart + h - hstart) * this->kernel_size_w_ + (kwstart + w - wstart)];
                //                 }
                //             }
                //             if (this->bias_term_)
                //             {
                //                 aveval += bias_data[c];
                //             }
                //             top_data[i] = aveval;
                //         }
                //     }
                //     break;
                // case memory::NHWC:
                //     tops[0].reset(new memory::tensor<float>(std::vector<int>{this->num_, this->output_dim_h_, this->output_dim_w_, this->output_channel_},
                //                                             bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
                //     NOT_IMPLEMENTED;
                //     break;
                // default:
                //     NOT_IMPLEMENTED;
                //     break;
                // }
            }

            this->suffix_activation_cpu_f32(tops);
        }

        template <typename Dtype>
        void operation_convolutiondepthwise<Dtype>::forward_cpu_i8(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms,
                                                                   std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            if (this->int8_scale_term_ == 1)
            {
                dequantize_int8(this->weights_i8_[0], this->weights_f32_[0], this->weights_scaletable_i8_);
                this->int8_scale_term_ = 0;
            }

            forward_cpu_f32(bottoms, tops);
        }

        template <typename Dtype>
        int operation_convolutiondepthwise<Dtype>::dequantize_int8(const std::shared_ptr<memory::tensor<signed char>> &src,
                                                                   std::shared_ptr<memory::tensor<float>> &dst, std::vector<float> scale)
        {
            size_t align_data_size = (this->weight_data_size_ + 4 - 1) & -4;
            int w = src->width();
            int size = align_data_size / this->group_;
            dst.reset(new memory::tensor<float>(align_data_size, this->params_.device_, src->order(), nullptr));
            const signed char *bottom = src->cpu_data();
            float *bottom_int8 = dst->mutable_cpu_data();

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
            for (int q = 0; q < this->output_channel_; q++)
            {
                float _scale = scale[q] <= 1e-6 ? 0 : 1.f / scale[q];
                const signed char *ptr = bottom + q * size;
                float *outptr = bottom_int8 + q * size;
                for (int i = 0; i < size; i++)
                {
                    outptr[i] = ptr[i] * _scale;
                }
            }
            return 0;
        }

        template <typename Dtype>
        void operation_convolutiondepthwise<Dtype>::forward_winograd_f32(std::shared_ptr<memory::tensor<float>> &bottom,
                                                                         std::shared_ptr<memory::tensor<float>> &top)
        {
            if (bottom->order() != memory::NCHW)
            {
                bottom->convert_order();
            }
            top.reset(new memory::tensor<float>(std::vector<int>{this->num_, this->output_channel_, this->output_dim_h_, this->output_dim_w_},
                                                bottom->device(), bottom->order(), bottom->allocator()));
            std::shared_ptr<memory::tensor<float>> border_bottom;
            int wino_pad_h = (this->input_dim_h_ + this->pad_bottom_ + this->pad_top_) % 2;
            int wino_pad_w = (this->input_dim_w_ + this->pad_left_ + this->pad_right_) % 2;
            make_border<float>(bottom, border_bottom, this->pad_top_, this->pad_bottom_ + wino_pad_h, this->pad_left_, this->pad_right_ + wino_pad_w, border_constant, this->pad_value_);
            const int tile_height = (border_bottom->height() - 4) / 2 + 1;
            const int tile_width = (border_bottom->width() - 4) / 2 + 1;
            const int b_width = border_bottom->width();
            std::shared_ptr<memory::tensor<float>> border_top;
            border_top.reset(new memory::tensor<float>(std::vector<int>{this->num_, this->output_channel_, tile_height * 2, tile_width * 2},
                                                       top->device(), memory::NCHW, top->allocator()));
            const int t_width = border_top->width();
            const float *bottom_data = border_bottom->cpu_data();
            //const float* weights_data = this->weights_f32_[0]->cpu_data();
            const float *bias_data = nullptr;
            if (this->bias_term_)
            {
                bias_data = this->weights_f32_[1]->cpu_data();
            }
            const float *U = U_->cpu_data();
            float *V = V_->mutable_cpu_data();
            for (size_t n = 0; n < this->num_; n++)
            {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (int g = 0; g < this->group_; g++)
                {
                    const float *bottom_data_slice = bottom_data + g * border_bottom->count(2, 4);
                    float *top_data_slice = border_top->mutable_cpu_data() + g * border_top->count(2, 4);
                    const float *u = U + 16 * g;
                    float *v = V + 16 * g;
                    float d[16];
                    for (size_t tile_h = 0; tile_h < tile_height; tile_h++)
                    {
                        for (size_t tile_w = 0; tile_w < tile_width; tile_w++)
                        {
                            //get d
                            for (size_t i = 0; i < 16; i++)
                            {
                                d[i] = bottom_data_slice[(tile_h * 2 + i / 4) * b_width + tile_w * 2 + i % 4];
                            }
                            //F(2,3), caclulate B^TdB
                            v[0] = d[0];
                            v[1] = d[1] - d[2] - d[3];
                            v[2] = -d[0] + d[1] + d[2];
                            v[3] = -d[3];
                            v[4] = d[4] - d[8] + d[12];
                            v[5] = d[5] - d[6] + d[7] - d[9] + d[10] - d[11] + d[13] - d[14] + d[15];
                            v[6] = -d[4] + d[5] + d[6] + d[8] - d[9] - d[10] - d[12] + d[13] + d[14];
                            v[7] = -d[7] + d[11] - d[15];
                            v[8] = -d[0] + d[4] + d[8];
                            v[9] = -d[1] + d[2] - d[3] + d[5] - d[6] + d[7] + d[9] - d[10] + d[11];
                            v[10] = d[0] - d[1] - d[2] - d[4] + d[5] + d[6] - d[8] + d[9] + d[10];
                            v[11] = d[3] - d[7] - d[11];
                            v[12] = -d[12];
                            v[13] = -d[13] + d[14] - d[15];
                            v[14] = d[12] - d[13] - d[14];
                            v[15] = d[15];
                            //calculate M = U.*V
                            for (size_t i = 0; i < 16 / mm_align_size; i++)
                            {
#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION)
                                __m256 vx = _mm256_load_ps(v + i * mm_align_size);
                                __m256 ux = _mm256_load_ps(u + i * mm_align_size);
                                vx = _mm256_mul_ps(vx, ux);
                                _mm256_store_ps(v + i * mm_align_size, vx);
#else
                                NOT_IMPLEMENTED;
#endif
                            }
                            //calculate A^TMA
                            top_data_slice[(tile_h * 2) * t_width + tile_w * 2] = v[0] + v[1] + v[2] + v[4] + v[5] + v[6] + v[8] + v[9] + v[10] + this->bias_term_ ? bias_data[g] : 0.0f;
                            top_data_slice[(tile_h * 2) * t_width + tile_w * 2 + 1] = v[1] - v[2] + v[3] + v[5] - v[6] + v[7] + v[9] - v[10] + v[11] + this->bias_term_ ? bias_data[g] : 0.0f;
                            top_data_slice[(tile_h * 2 + 1) * t_width + tile_w * 2] = v[4] + v[5] + v[6] - v[8] - v[9] - v[10] + v[12] + v[13] + v[14] + this->bias_term_ ? bias_data[g] : 0.0f;
                            top_data_slice[(tile_h * 2 + 1) * t_width + tile_w * 2 + 1] = v[5] - v[6] + v[7] - v[9] + v[10] - v[11] + v[13] - v[14] + v[15] + this->bias_term_ ? bias_data[g] : 0.0f;
                        }
                    }
                }
            }
            //cut pad border
            for (size_t h = 0; h < this->output_dim_h_; h++)
            {
                memcpy(top->mutable_cpu_data() + h * this->output_dim_w_, border_top->cpu_data() + h * (this->output_dim_w_ + wino_pad_w), this->output_dim_w_ * sizeof(float));
            }
        }

        template <typename Dtype>
        void operation_convolutiondepthwise<Dtype>::forward_k3s1_f32(const std::shared_ptr<memory::tensor<float>> &bottom,
                                                                     std::shared_ptr<memory::tensor<float>> &top)
        {
            std::shared_ptr<memory::tensor<float>> bottom_bordered;
            make_border<float>(bottom, bottom_bordered, this->pad_top_, this->pad_bottom_, this->pad_left_, this->pad_right_, border_constant, this->pad_value_);
            int inw = bottom_bordered->width();
            if (bottom_bordered->order() != memory::NCHW)
            {
                bottom_bordered->convert_order();
            }
            top.reset(new memory::tensor<float>(std::vector<int>{this->num_, this->output_channel_, this->output_dim_h_, this->output_dim_w_},
                                                bottom->device(), bottom->order(), bottom->allocator()));

            const int top_cstep = top->count(2, 4);
            const int bottom_cstep = bottom_bordered->count(2, 4);
            const float *weights_data = this->weights_f32_[0]->cpu_data();
            const float *bias_data = nullptr;
            if (this->bias_term_)
            {
                bias_data = this->weights_f32_[1]->cpu_data();
            }

            for (int n = 0; n < this->num_; n++)
            {
                float *top_data = top->mutable_cpu_data() + n * top->count(1, 4);
                auto bottom_data = bottom_bordered->cpu_data() + n * bottom_bordered->count(1, 4);
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                for (int g = 0; g < this->group_; g++)
                {
                    float *out = top_data + g * top_cstep;
                    const float bias_data0 = this->bias_term_ ? bias_data[g] : 0.f;
                    const float *weights_data0 = weights_data + g * 9;
                    float *outptr = out;
                    float *outptr2 = outptr + this->output_dim_w_;
                    const float *img0 = bottom_data + g * bottom_cstep;

                    const float *r0 = img0;
                    const float *r1 = img0 + inw;
                    const float *r2 = img0 + inw * 2;
                    const float *r3 = img0 + inw * 3;

                    const float *k0 = weights_data0;
                    const float *k1 = weights_data0 + 3;
                    const float *k2 = weights_data0 + 6;

#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX //AVX
                    __m128 k0_data = _mm_loadu_ps(k0);
                    __m128 k1_data = _mm_loadu_ps(k1);
                    __m128 k2_data = _mm_loadu_ps(k2);
#endif
                    int i = 0;
                    for (; i + 1 < this->output_dim_h_; i += 2)
                    {
                        int remain = this->output_dim_w_;
                        for (; remain > 0; remain--)
                        {
                            float sum_sum = bias_data0;
                            float sum_sum2 = bias_data0;

#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX //AVX
                            __m128 sum = _mm_setzero_ps();
                            __m128 sum2 = _mm_setzero_ps();
                            __m128 r0_data = _mm_loadu_ps(r0);
                            __m128 r1_data = _mm_loadu_ps(r1);
                            __m128 r2_data = _mm_loadu_ps(r2);
                            __m128 r3_data = _mm_loadu_ps(r3);

                            sum = _mm_fmadd_ps(r0_data, k0_data, sum);
                            sum = _mm_fmadd_ps(r1_data, k1_data, sum);
                            sum = _mm_fmadd_ps(r2_data, k2_data, sum);
                            //sum_sum += sum.m128_f32[0] + sum.m128_f32[1] + sum.m128_f32[2];

                            float temp[4];
                            _mm_storeu_ps(temp, sum);
                            for (int i = 0; i < 3; i++)
                            {
                                sum_sum += temp[i];
                            }

                            sum2 = _mm_fmadd_ps(r1_data, k0_data, sum2);
                            sum2 = _mm_fmadd_ps(r2_data, k1_data, sum2);
                            sum2 = _mm_fmadd_ps(r3_data, k2_data, sum2);

                            float temp2[4];
                            _mm_storeu_ps(temp2, sum2);
                            for (int i = 0; i < 3; i++)
                            {
                                sum_sum2 += temp2[i];
                            }
#else
                            sum_sum += r0[0] * k0[0];
                            sum_sum += r0[1] * k0[1];
                            sum_sum += r0[2] * k0[2];
                            sum_sum += r1[0] * k1[0];
                            sum_sum += r1[1] * k1[1];
                            sum_sum += r1[2] * k1[2];
                            sum_sum += r2[0] * k2[0];
                            sum_sum += r2[1] * k2[1];
                            sum_sum += r2[2] * k2[2];

                            sum_sum2 += r1[0] * k0[0];
                            sum_sum2 += r1[1] * k0[1];
                            sum_sum2 += r1[2] * k0[2];
                            sum_sum2 += r2[0] * k1[0];
                            sum_sum2 += r2[1] * k1[1];
                            sum_sum2 += r2[2] * k1[2];
                            sum_sum2 += r3[0] * k2[0];
                            sum_sum2 += r3[1] * k2[1];
                            sum_sum2 += r3[2] * k2[2];
#endif

                            *outptr += sum_sum;
                            *outptr2 += sum_sum2;

                            r0++;
                            r1++;
                            r2++;
                            r3++;
                            outptr++;
                            outptr2++;
                        }

                        r0 += 2 + inw;
                        r1 += 2 + inw;
                        r2 += 2 + inw;
                        r3 += 2 + inw;

                        outptr += this->output_dim_w_;
                        outptr2 += this->output_dim_w_;
                    }

                    for (; i < this->output_dim_h_; i++)
                    {
                        int remain = this->output_dim_w_;

                        for (; remain > 0; remain--)
                        {
                            float sum_sum = bias_data0;
#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX //AVX
                            __m128 sum = _mm_setzero_ps();

                            __m128 r0_data = _mm_loadu_ps(r0);
                            __m128 r1_data = _mm_loadu_ps(r1);
                            __m128 r2_data = _mm_loadu_ps(r2);
                            __m128 r3_data = _mm_loadu_ps(r3);

                            sum = _mm_fmadd_ps(r0_data, k0_data, sum);
                            sum = _mm_fmadd_ps(r1_data, k1_data, sum);
                            sum = _mm_fmadd_ps(r2_data, k2_data, sum);

                            float temp[4];
                            _mm_storeu_ps(temp, sum);
                            for (int i = 0; i < 3; i++)
                            {
                                sum_sum += temp[i];
                            }
#else
                            sum_sum += r0[0] * k0[0];
                            sum_sum += r0[1] * k0[1];
                            sum_sum += r0[2] * k0[2];
                            sum_sum += r1[0] * k1[0];
                            sum_sum += r1[1] * k1[1];
                            sum_sum += r1[2] * k1[2];
                            sum_sum += r2[0] * k2[0];
                            sum_sum += r2[1] * k2[1];
                            sum_sum += r2[2] * k2[2];
#endif
                            *outptr += sum_sum;
                            r0++;
                            r1++;
                            r2++;
                            outptr++;
                        }

                        r0 += 2;
                        r1 += 2;
                        r2 += 2;
                    }
                }
            }
        }

        template <typename Dtype>
        void operation_convolutiondepthwise<Dtype>::forward_k3s2_f32(const std::shared_ptr<memory::tensor<float>> &bottom,
                                                                     std::shared_ptr<memory::tensor<float>> &top)
        {
            profiler *p = profiler::get();
            p->scope_start("make_border");
            std::shared_ptr<memory::tensor<float>> bottom_bordered;
            make_border<float>(bottom, bottom_bordered, this->pad_top_, this->pad_bottom_, this->pad_left_, this->pad_right_, border_constant, this->pad_value_);
            if (bottom_bordered->order() != memory::NCHW)
            {
                bottom_bordered->convert_order();
            }
            p->scope_end();
            p->scope_start("reset");
            top.reset(new memory::tensor<float>(std::vector<int>{this->num_, this->output_channel_, this->output_dim_h_, this->output_dim_w_},
                                                bottom->device(), bottom->order(), bottom->allocator()));
            const int top_cstep = top->count(2, 4);
            const int bottom_cstep = bottom_bordered->count(2, 4);
            const float *weights_data = this->weights_f32_[0]->cpu_data();
            const float *bias_data = nullptr;
            if (this->bias_term_)
            {
                bias_data = this->weights_f32_[1]->cpu_data();
            }
            const int tailstep = bottom_bordered->width() - 2 * this->output_dim_w_ + bottom_bordered->width();
            p->scope_end();
            p->scope_start("exec");

            for (int n = 0; n < this->num_; n++)
            {
                float *top_data = top->mutable_cpu_data() + n * top->count(1, 4);
                auto bottom_data = bottom_bordered->cpu_data() + n * bottom_bordered->count(1, 4);
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                for (int g = 0; g < this->group_; g++)
                {
                    float *out = top_data + g * top_cstep;

                    const float bias0 = this->bias_term_ ? bias_data[g] : 0.f;

                    const float *kernel0 = weights_data + g * 9;

                    float *outptr = out;

                    const float *img0 = bottom_data + g * bottom_cstep;

                    const float *r0 = img0;
                    const float *r1 = img0 + bottom_bordered->width();
                    const float *r2 = img0 + bottom_bordered->width() * 2;

                    const float *k0 = kernel0;
                    const float *k1 = kernel0 + 3;
                    const float *k2 = kernel0 + 6;
#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX //AVX
                    __m128 k0_data = _mm_loadu_ps(k0);
                    __m128 k1_data = _mm_loadu_ps(k1);
                    __m128 k2_data = _mm_loadu_ps(k2);

                    int i = 0;

                    for (; i < this->output_dim_h_; i++)
                    {
                        int remain = this->output_dim_w_;

                        for (; remain > 0; remain--)
                        {
                            float sum_sum = bias0;
                            __m128 sum = _mm_setzero_ps();
                            __m128 r0_data = _mm_loadu_ps(r0);
                            __m128 r1_data = _mm_loadu_ps(r1);
                            __m128 r2_data = _mm_loadu_ps(r2);

                            sum = _mm_fmadd_ps(r0_data, k0_data, sum);
                            sum = _mm_fmadd_ps(r1_data, k1_data, sum);
                            sum = _mm_fmadd_ps(r2_data, k2_data, sum);
                            //sum_sum += sum.m128_f32[0] + sum.m128_f32[1] + sum.m128_f32[2];

                            float temp[4];
                            _mm_storeu_ps(temp, sum);
                            for (int i = 0; i < 3; i++)
                            {
                                sum_sum += temp[i];
                            }
                            *outptr += sum_sum;
                            r0 += 2;
                            r1 += 2;
                            r2 += 2;
                            outptr++;
                        }

                        r0 += tailstep;
                        r1 += tailstep;
                        r2 += tailstep;
                    }
#else
                    int i = 0;

                    for (; i < this->output_dim_h_; i++)
                    {
                        int remain = this->output_dim_w_;

                        for (; remain > 0; remain--)
                        {
                            *outptr = mul_add_3x3_native(r0, r1, r2, k0, k1, k2, bias0);

                            r0 += 2;
                            r1 += 2;
                            r2 += 2;
                            outptr++;
                        }

                        r0 += tailstep;
                        r1 += tailstep;
                        r2 += tailstep;
                    }
#endif
                }
            }

            p->scope_end();
        }

#ifndef USE_CUDA
        STUB_GPU(operation_convolutiondepthwise);
#endif

        INSTANCE_CLASS(operation_convolutiondepthwise);
        REGISTE(operation_convolutiondepthwise);
    }
}