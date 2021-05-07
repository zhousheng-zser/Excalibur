#include "../../include/Excalibur/operation_convolution.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"
#include "../../include/Excalibur/im2col.hpp"
#include "../../include/Excalibur/math_functions.hpp"
#include "../../include/Excalibur/operation_make_border.hpp"
#include "../../include/Excalibur/operation_cut_border.hpp"
#include "../../include/Primitives/simd_types.hpp"
#include <random>

namespace glasssix
{
    namespace excalibur
    {
        template <typename Dtype>
        operation_convolution<Dtype>::operation_convolution(const operation_param &param) : operation_general_conv<Dtype>(param)
        {
        }

        template <typename Dtype>
        int operation_convolution<Dtype>::init_weights(FILE *fp)
        {
            int quantize_tag;
            fread(&quantize_tag, 1, sizeof(int), fp);
            int mem = 0;
            if (quantize_tag == 0)
            {
                this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(this->weight_data_size_, this->params_.device_, memory::NCHW, nullptr)));
                fread(this->weights_f32_[0]->mutable_cpu_data(), 1, this->weight_data_size_ * sizeof(float), fp);
                mem += this->weight_data_size_ * sizeof(float);
                if (this->bias_term_)
                {
                    this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(this->output_channel_, this->params_.device_, memory::NCHW, nullptr)));
                    fread(this->weights_f32_[1]->mutable_cpu_data(), 1, this->output_channel_ * sizeof(float), fp);
                    mem += this->output_channel_ * sizeof(float);
                }

                if (this->params_.device_ < 0)
                {
                    if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 1 && this->stride_w_ == 1) && this->output_channel_ < 128)
                    {
                        conv3x3s1_winograd23_tr_kernel();
                    }
                    else
                    {
                        forward_im2col_tr_kernel();
                    }
                }
            }
            else if (quantize_tag == 871224)
            {
                size_t align_data_size = (this->weight_data_size_ + 4 - 1) & -4;
                this->weights_i8_.push_back(std::shared_ptr<memory::tensor<signed char>>(new memory::tensor<signed char>(align_data_size, this->params_.device_, memory::NCHW, nullptr)));
                fread(this->weights_i8_[0]->mutable_cpu_data(), 1, align_data_size, fp);
                mem += align_data_size;
                if (this->bias_term_)
                {
                    this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(1, this->params_.device_, memory::NCHW, nullptr)));
                    this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(this->output_channel_, this->params_.device_, memory::NCHW, nullptr)));
                    fread(this->weights_f32_[1]->mutable_cpu_data(), 1, this->output_channel_ * sizeof(float), fp);
                    mem += this->output_channel_ * sizeof(float);
                }
                this->weights_scaletable_i8_.resize(this->output_channel_);
                fread(this->weights_scaletable_i8_.data(), 1, this->output_channel_ * sizeof(float), fp);
                this->featmap_scaletable_i8_.resize(1);
                fread(this->featmap_scaletable_i8_.data(), 1, 1 * sizeof(float), fp);
                mem += (this->output_channel_ + 1) * sizeof(float);

                if (this->params_.device_ < 0)
                {
                    if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 1 && this->stride_w_ == 1) && this->output_channel_ < 128)
                    {
                        conv3x3s1_winograd23_tr_kernel_int8();
                    }
                    else
                    {
                        forward_im2col_int8_tr_kernel();
                    }
                }
            }
            else
            {
                NOT_IMPLEMENTED;
            }
            return mem;
        }

        template <typename Dtype>
        int operation_convolution<Dtype>::init_weights()
        {
            std::default_random_engine e;
            std::normal_distribution<float> n(0, 0.3);
            std::uniform_int_distribution<int> u(-128, 127);
            int mem = 0;
            if (!this->params_.int8_quantization_)
            {
                this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(this->weight_data_size_, this->params_.device_, memory::NCHW, nullptr)));
                for (size_t i = 0; i < this->weight_data_size_; i++)
                {
                    this->weights_f32_[0]->mutable_cpu_data()[i] = n(e);
                }
                mem += this->weight_data_size_ * sizeof(float);
                if (this->bias_term_)
                {
                    this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(this->output_channel_, this->params_.device_, memory::NCHW, nullptr)));
                    for (size_t i = 0; i < this->output_channel_; i++)
                    {
                        this->weights_f32_[1]->mutable_cpu_data()[i] = n(e);
                    }
                    mem += this->output_channel_ * sizeof(float);
                }

                if (this->params_.device_)
                {
                    if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 1 && this->stride_w_ == 1) && this->output_channel_ < 128)
                    {
                        conv3x3s1_winograd23_tr_kernel();
                    }
                    else
                    {
                        forward_im2col_tr_kernel();
                    }
                }
            }
            else
            {
                size_t align_data_size = (this->weight_data_size_ + 4 - 1) & -4;
                this->weights_i8_.push_back(std::shared_ptr<memory::tensor<signed char>>(new memory::tensor<signed char>(align_data_size, this->params_.device_, memory::NCHW, nullptr)));
                for (size_t i = 0; i < align_data_size; i++)
                {
                    this->weights_i8_[0]->mutable_cpu_data()[i] = u(e);
                }
                mem += align_data_size;
                if (this->bias_term_)
                {
                    this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(1, this->params_.device_, memory::NCHW, nullptr)));
                    this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(this->output_channel_, this->params_.device_, memory::NCHW, nullptr)));
                    for (size_t i = 0; i < this->output_channel_; i++)
                    {
                        this->weights_f32_[1]->mutable_cpu_data()[i] = n(e);
                    }
                    mem += this->output_channel_ * sizeof(float);
                }
                this->weights_scaletable_i8_.resize(this->output_channel_);
                for (size_t i = 0; i < this->output_channel_; i++)
                {
                    this->weights_scaletable_i8_[i] = n(e);
                }
                this->featmap_scaletable_i8_.resize(1);
                this->featmap_scaletable_i8_[0] = n(e);
                mem += (this->output_channel_ + 1) * sizeof(float);

                if (this->params_.device_)
                {
                    if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 1 && this->stride_w_ == 1) && this->output_channel_ < 128)
                    {
                        conv3x3s1_winograd23_tr_kernel_int8();
                    }
                    else
                    {
                        forward_im2col_int8_tr_kernel();
                    }
                }
            }
            return mem;
        }

        template <typename Dtype>
        void operation_convolution<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms,
                                                           std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);
            memory::orderType order = bottoms[0]->order();
            this->num_ = bottoms[0]->num();
            const float *bottom_data = bottoms[0]->cpu_data();
            const float *weights_data = this->weights_f32_[0]->cpu_data();
            const float *bias_data = nullptr;

            this->input_channel_ = bottoms[0]->channels();
            this->input_dim_h_ = bottoms[0]->height();
            this->input_dim_w_ = bottoms[0]->width();
            this->output_dim_h_ = (this->input_dim_h_ + this->pad_bottom_ + this->pad_top_ - this->kernel_size_h_) / this->stride_h_ + 1;
            this->output_dim_w_ = (this->input_dim_w_ + this->pad_left_ + this->pad_right_ - this->kernel_size_w_) / this->stride_w_ + 1;
            this->output_spatial_dim_ = this->output_dim_w_ * this->output_dim_h_;
            this->kernel_dim_ = this->input_channel_ * this->kernel_size_h_ * this->kernel_size_w_;

            if (order == memory::NCHW)
            {
                tops[0].reset(new memory::tensor<float>(std::vector<int>{this->num_, this->output_channel_, this->output_dim_h_, this->output_dim_w_}, this->params_.device_, order, bottoms[0]->allocator()));
            }
            else if (order == memory::NHWC)
            {
                tops[0].reset(new memory::tensor<float>(std::vector<int>{this->num_, this->output_dim_h_, this->output_dim_w_, this->output_channel_}, this->params_.device_, order, bottoms[0]->allocator()));
            }
            else
            {
                LOG(FATAL) << "Un-supported data arrange.";
            }
            float *top_data = tops[0]->mutable_cpu_data();
            col_buffer_.reset(new memory::tensor<float>(std::vector<int>{this->kernel_dim_ / this->group_, this->output_dim_h_, this->output_dim_w_}, this->params_.device_, memory::NCHW, bottoms[0]->allocator()));
            col_buffer_data = col_buffer_->mutable_cpu_data();
            if (this->bias_term_)
            {
                bias_multiplier_.reset(new memory::tensor<float>(std::vector<int>{this->output_dim_w_ * this->output_dim_h_}, this->params_.device_, memory::NCHW, bottoms[0]->allocator()));
                bias_multiplier_data = bias_multiplier_->mutable_cpu_data();
                math_functions::cpu_set(this->output_spatial_dim_, 1.0f, bias_multiplier_data);
                bias_data = this->weights_f32_[1]->cpu_data();
            }

            if (this->pad_left_ != 0)
            {
                make_border<float>(bottoms[0], border_bottom_, this->pad_top_, this->pad_bottom_, this->pad_left_, this->pad_right_, border_constant, this->pad_value_);
            }
            else
            {
                border_bottom_ = bottoms[0];
            }
            // k3s1
            if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 1 && this->stride_w_ == 1) && this->output_channel_ < 128)
            {
                conv3x3s1_winograd23(border_bottom_, tops[0]);
            }
            else
            {
                forward_im2col(border_bottom_, tops[0]);
            }
            this->suffix_activation_cpu_f32(tops);
        }

        template <typename Dtype>
        void operation_convolution<Dtype>::forward_cpu_i8(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms,
                                                          std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);
            memory::orderType order = bottoms[0]->order();
            this->num_ = bottoms[0]->num();
            const float *bottom_data = bottoms[0]->cpu_data();
            const signed char *weights_data = this->weights_i8_[0]->cpu_data();
            const float *bias_data = nullptr;
            this->input_channel_ = bottoms[0]->channels();
            this->input_dim_h_ = bottoms[0]->height();
            this->input_dim_w_ = bottoms[0]->width();

            bool use_winograd3x3_int8;

            this->output_dim_h_ = (this->input_dim_h_ + this->pad_bottom_ + this->pad_top_ - this->kernel_size_h_) / this->stride_h_ + 1;
            this->output_dim_w_ = (this->input_dim_w_ + this->pad_left_ + this->pad_right_ - this->kernel_size_w_) / this->stride_w_ + 1;
            this->output_spatial_dim_ = this->output_dim_w_ * this->output_dim_h_;
            if (order == memory::NCHW)
            {
                tops[0].reset(new memory::tensor<float>(std::vector<int>{this->num_, this->output_channel_, this->output_dim_h_, this->output_dim_w_}, this->params_.device_, order, bottoms[0]->allocator()));
                top_int32_.reset(new memory::tensor<int>(std::vector<int>{this->num_, this->output_channel_, this->output_dim_h_, this->output_dim_w_}, this->params_.device_, order, nullptr));
            }
            else if (order == memory::NHWC)
            {
                tops[0].reset(new memory::tensor<float>(std::vector<int>{this->num_, this->output_dim_h_, this->output_dim_w_, this->output_channel_}, this->params_.device_, order, bottoms[0]->allocator()));
                top_int32_.reset(new memory::tensor<int>(std::vector<int>{this->num_, this->output_channel_, this->output_dim_h_, this->output_dim_w_}, this->params_.device_, order, nullptr));
            }
            else
            {
                LOG(FATAL) << "Un-supported data arrange.";
            }
            float *top_data = tops[0]->mutable_cpu_data();
            if (this->bias_term_)
            {
                bias_data = this->weights_f32_[1]->cpu_data();
            }
            use_winograd3x3_int8 = false;
            this->quantize_float32_to_int8(bottoms[0], bottom_int8_);

            if (this->pad_left_ != 0 || this->pad_right_ != 0 || this->pad_top_ != 0 || this->pad_bottom_ != 0)
            {
                make_border<signed char>(bottom_int8_, bottom_int8_bordered_, this->pad_top_, this->pad_bottom_, this->pad_left_, this->pad_right_, border_constant, this->pad_value_);
            }
            else
            {
                bottom_int8_bordered_ = bottom_int8_;
            }
            if ((this->kernel_size_h_ == 3 && this->kernel_size_w_ == 3) && (this->stride_h_ == 1 && this->stride_w_ == 1) && this->output_channel_ < 128)
            {
                // winograd is slow on small channel count
                use_winograd3x3_int8 = true;
                conv3x3s1_winograd23_int8(bottom_int8_bordered_, top_int32_);

                this->dequantize_int32_to_float32(top_int32_, tops[0]);
            }
            else
            {
                std::vector<float> dequantize_scales;
                for (size_t q = 0; q < this->output_channel_; q++)
                {
                    float scale_in;
                    if (this->weights_scaletable_i8_[q] <= 1e-6)
                        scale_in = 0;
                    else
                        scale_in = 1.f / (this->featmap_scaletable_i8_[0] * this->weights_scaletable_i8_[q]);
                    dequantize_scales.push_back(scale_in);
                }

                conv_im2col_sgemm_int8_dequant_sse(bottom_int8_bordered_, tops[0], dequantize_scales);
            }
            this->suffix_activation_cpu_f32(tops);
        }

        template <typename Dtype>
        void operation_convolution<Dtype>::conv3x3s1_winograd23_tr_kernel()
        {
            const int input_channel_ = this->weight_data_size_ * this->group_ / this->output_channel_ / this->kernel_size_h_ / this->kernel_size_w_;
            int inch = input_channel_ / this->group_;
            int outch = this->output_channel_ / this->group_;

            const float *kernel_data_base = this->weights_f32_[0]->cpu_data();
            // G
            const float ktm[4][3] = {
                {1.0f, 0.0f, 0.0f},
                {1.0f / 2, 1.0f / 2, 1.0f / 2},
                {1.0f / 2, -1.0f / 2, 1.0f / 2},
                {0.0f, 0.0f, 1.0f}};

            for (int g = 0; g < this->group_; g++)
            {
                kernel_tm_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(std::vector<int>{1, outch, inch, 4 * 4}, this->params_.device_, memory::NCHW, nullptr)));
                float *kernel_tm_data = kernel_tm_[g]->mutable_cpu_data();
                int kernel_tm_w = kernel_tm_[g]->width();
                int kernel_tm_h = kernel_tm_[g]->height();
                int kernel_tm_cstep = kernel_tm_w * kernel_tm_h;
                const float *kernel_data = kernel_data_base + this->weight_data_size_ * g / this->group_;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                for (int p = 0; p < outch; p++)
                {
                    for (int q = 0; q < inch; q++)
                    {
                        // transform kernel
                        const float *kernel0 = kernel_data + p * inch * 9 + q * 9;
                        float *kernel_tm0 = kernel_tm_data + p * kernel_tm_cstep + q * kernel_tm_w;
                        const float *k0 = kernel0;
                        const float *k1 = kernel0 + 3;
                        const float *k2 = kernel0 + 6;

                        // tmp = G * (gT)T = G*g
                        float tmp[4][3];
                        for (int i = 0; i < 4; i++)
                        {
                            tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                            tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                            tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
                        }
                        // U
                        for (int j = 0; j < 4; j++)
                        {
                            float *tmpp = &tmp[j][0];
                            for (int i = 0; i < 4; i++)
                                kernel_tm0[j * 4 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                        }
                    }
                }
            }
        }

        template <typename Dtype>
        void operation_convolution<Dtype>::conv3x3s1_winograd23(const std::shared_ptr<memory::tensor<float>> &bottom_blob,
                                                                std::shared_ptr<memory::tensor<float>> &top_blob)
        {
            int num = bottom_blob->num();
            int w = bottom_blob->width();
            int h = bottom_blob->height();
            int inch = bottom_blob->channels() / this->group_;
            top_blob.reset(new memory::tensor<float>(std::vector<int>{num, this->output_channel_, h - 2, w - 2}, -1, memory::NCHW, bottom_blob->allocator()));

            int outw = top_blob->width();
            int outh = top_blob->height();
            int ou = top_blob->channels() / this->group_;

            // pad to 2n+2, winograd F(2,3)
            outw = (outw + 1) >> 1 << 1;
            outh = (outh + 1) >> 1 << 1;
            w = outw + 2;
            h = outh + 2;
            bottom_blob_bordered_ = bottom_blob;
            make_border<float>(bottom_blob, bottom_blob_bordered_, 0, h - bottom_blob->height(), 0, w - bottom_blob->width(), border_constant, 0);

            float *bottom_blob_bordered_data = bottom_blob_bordered_->mutable_cpu_data();
            int bordered_h = bottom_blob_bordered_->height();
            int bordered_w = bottom_blob_bordered_->width();
            const float *bias_0 = nullptr;
            if (this->bias_term_)
            {
                bias_0 = this->weights_f32_[1]->cpu_data();
            }

            top_blob_bordered_.reset(new memory::tensor<float>(std::vector<int>{num, this->output_channel_, outh, outw}, -1, memory::NCHW, bottom_blob->allocator()));
            float *top_data_base = top_blob_bordered_->mutable_cpu_data();

            for (int n = 0; n < num; n++)
            {
                float *bottom_data_base = bottom_blob_bordered_data + n * inch * bordered_h * bordered_w * this->group_;
                for (int g = 0; g < this->group_; g++)
                {
                    const float *kernel_tm_data = kernel_tm_[g]->cpu_data();
                    const float *bottom_blob_bordered_data_n = bottom_data_base + g * inch * bordered_h * bordered_w;
                    const float *bias = bias_0 + g * ou;
                    float *top_blob_bordered_data = top_data_base + g * ou * outh * outw;

                    {
                        int w_tm = outw >> 1 << 2;
                        int h_tm = outh >> 1 << 2;
                        int block_col = h_tm >> 2;
                        int block_row = w_tm >> 2;

                        const int tiles = block_col * block_row;
                        bottom_blob_tm_.reset(new memory::tensor<float>(std::vector<int>{1, inch, tiles, 16}, -1, memory::NCHW, bottom_blob->allocator()));
                        float *bottom_blob_tm_data = bottom_blob_tm_->mutable_cpu_data();

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                        for (int q = 0; q < inch; q++)
                        {
                            const float *img = bottom_blob_bordered_data_n + q * bordered_h * bordered_w;
                            float *out_tm0 = bottom_blob_tm_data + q * 16 * tiles;

                            for (int j = 0; j < block_col; j++)
                            {
                                const float *r0 = img + w * j * 2;
                                const float *r1 = r0 + w;
                                const float *r2 = r1 + w;
                                const float *r3 = r2 + w;
                                for (int i = 0; i < block_row; i++)
                                {
#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX //AVX
                                    __m128 _d0, _d1, _d2, _d3;
                                    __m128 _w0, _w1, _w2, _w3;

                                    // load
                                    _d0 = _mm_loadu_ps(r0);
                                    _d1 = _mm_loadu_ps(r1);
                                    _d2 = _mm_loadu_ps(r2);
                                    _d3 = _mm_loadu_ps(r3);

                                    // w = B_t * d
                                    _w0 = _mm_sub_ps(_d0, _d2);
                                    _w1 = _mm_add_ps(_d1, _d2);
                                    _w2 = _mm_sub_ps(_d2, _d1);
                                    _w3 = _mm_sub_ps(_d3, _d1);

                                    // transpose d to d_t
                                    _MM_TRANSPOSE4_PS(_w0, _w1, _w2, _w3);

                                    //_mm256_madd_epi16_epi32()

                                    // d = B_t * d_t
                                    _d0 = _mm_sub_ps(_w0, _w2);
                                    _d1 = _mm_add_ps(_w1, _w2);
                                    _d2 = _mm_sub_ps(_w2, _w1);
                                    _d3 = _mm_sub_ps(_w3, _w1);

                                    // save to out_tm
                                    _mm_storeu_ps(out_tm0, _d0);
                                    _mm_storeu_ps(out_tm0 + 4, _d1);
                                    _mm_storeu_ps(out_tm0 + 8, _d2);
                                    _mm_storeu_ps(out_tm0 + 12, _d3);
#else
                                    float d0[4], d1[4], d2[4], d3[4];
                                    float w0[4], w1[4], w2[4], w3[4];
                                    float t0[4], t1[4], t2[4], t3[4];
                                    // load
#include "operation_cxx/op_wino23_tr_kernel.cxx"
#endif
                                    r0 += 2;
                                    r1 += 2;
                                    r2 += 2;
                                    r3 += 2;
                                    out_tm0 += 16;
                                }
                            }
                        }
                    }
                    // BEGIN dot
                    std::shared_ptr<memory::tensor<float>> top_blob_tm;
                    {
                        int w_tm = outw >> 1 << 2;
                        int h_tm = outh >> 1 << 2;

                        int block_col = h_tm >> 2; // may be the block num in Feathercnn
                        int block_row = w_tm >> 2;
                        const int tiles = block_col * block_row;
                        top_blob_tm.reset(new memory::tensor<float>(std::vector<int>{1, ou, tiles, 16}, -1, memory::NCHW, bottom_blob->allocator()));
                        float *top_blob_tm_data = top_blob_tm->mutable_cpu_data();
                        float *bottom_blob_tm_data = bottom_blob_tm_->mutable_cpu_data();

                        int nn_ou = ou >> 2;
                        int remain_ou_start = nn_ou << 2;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                        for (int pp = 0; pp < nn_ou; pp++)
                        {
                            int p = pp << 2;

                            float *out0_tm = top_blob_tm_data + p * 16 * tiles;
                            float *out1_tm = top_blob_tm_data + (p + 1) * 16 * tiles;
                            float *out2_tm = top_blob_tm_data + (p + 2) * 16 * tiles;
                            float *out3_tm = top_blob_tm_data + (p + 3) * 16 * tiles;

                            const float *kernel0_tm = kernel_tm_data + p * inch * 16;
                            const float *kernel1_tm = kernel_tm_data + (p + 1) * inch * 16;
                            const float *kernel2_tm = kernel_tm_data + (p + 2) * inch * 16;
                            const float *kernel3_tm = kernel_tm_data + (p + 3) * inch * 16;
                            for (int i = 0; i < tiles; i++)
                            {
                                float *output0_tm = out0_tm + i * 16;
                                float *output1_tm = out1_tm + i * 16;
                                float *output2_tm = out2_tm + i * 16;
                                float *output3_tm = out3_tm + i * 16;

#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX
                                float zero_val = 0.f;
                                __m256 _sum0 = _mm256_broadcast_ss(&zero_val);
                                __m256 _sum0n = _mm256_broadcast_ss(&zero_val);
                                __m256 _sum1 = _mm256_broadcast_ss(&zero_val);
                                __m256 _sum1n = _mm256_broadcast_ss(&zero_val);
                                __m256 _sum2 = _mm256_broadcast_ss(&zero_val);
                                __m256 _sum2n = _mm256_broadcast_ss(&zero_val);
                                __m256 _sum3 = _mm256_broadcast_ss(&zero_val);
                                __m256 _sum3n = _mm256_broadcast_ss(&zero_val);

                                int q = 0;

#include "operation_cxx/op_wino23_sse1.cxx"

                                for (; q < inch; q++)
                                {
                                    const float *r0 = bottom_blob_tm_data + q * tiles * 16 + i * 16;
                                    const float *k0 = kernel0_tm + q * 16;
                                    const float *k1 = kernel1_tm + q * 16;
                                    const float *k2 = kernel2_tm + q * 16;
                                    const float *k3 = kernel3_tm + q * 16;

                                    __m256 _r0 = _mm256_loadu_ps(r0);
                                    __m256 _r0n = _mm256_loadu_ps(r0 + 8);
                                    __m256 _k0 = _mm256_loadu_ps(k0);
                                    __m256 _k0n = _mm256_loadu_ps(k0 + 8);
                                    __m256 _k1 = _mm256_loadu_ps(k1);
                                    __m256 _k1n = _mm256_loadu_ps(k1 + 8);
                                    __m256 _k2 = _mm256_loadu_ps(k2);
                                    __m256 _k2n = _mm256_loadu_ps(k2 + 8);
                                    __m256 _k3 = _mm256_loadu_ps(k3);
                                    __m256 _k3n = _mm256_loadu_ps(k3 + 8);

                                    _sum0 = _mm256_fmadd_ps(_r0, _k0, _sum0);
                                    _sum0n = _mm256_fmadd_ps(_r0n, _k0n, _sum0n);
                                    _sum1 = _mm256_fmadd_ps(_r0, _k1, _sum1);
                                    _sum1n = _mm256_fmadd_ps(_r0n, _k1n, _sum1n);
                                    _sum2 = _mm256_fmadd_ps(_r0, _k2, _sum2);
                                    _sum2n = _mm256_fmadd_ps(_r0n, _k2n, _sum2n);
                                    _sum3 = _mm256_fmadd_ps(_r0, _k3, _sum3);
                                    _sum3n = _mm256_fmadd_ps(_r0n, _k3n, _sum3n);
                                }

                                _mm256_storeu_ps(output0_tm, _sum0);
                                _mm256_storeu_ps(output0_tm + 8, _sum0n);
                                _mm256_storeu_ps(output1_tm, _sum1);
                                _mm256_storeu_ps(output1_tm + 8, _sum1n);
                                _mm256_storeu_ps(output2_tm, _sum2);
                                _mm256_storeu_ps(output2_tm + 8, _sum2n);
                                _mm256_storeu_ps(output3_tm, _sum3);
                                _mm256_storeu_ps(output3_tm + 8, _sum3n);
#else
                                float sum0[16] = {0.0f};
                                float sum1[16] = {0.0f};
                                float sum2[16] = {0.0f};
                                float sum3[16] = {0.0f};

                                int q = 0;
                                for (; q + 3 < inch; q += 4)
                                {
                                    const float *r0 = bottom_blob_tm_data + q * tiles * 16 + i * 16;
                                    const float *r1 = bottom_blob_tm_data + (q + 1) * tiles * 16 + i * 16;
                                    const float *r2 = bottom_blob_tm_data + (q + 2) * tiles * 16 + i * 16;
                                    const float *r3 = bottom_blob_tm_data + (q + 3) * tiles * 16 + i * 16;

                                    const float *k0 = kernel0_tm + q * 16;
                                    const float *k1 = kernel1_tm + q * 16;
                                    const float *k2 = kernel2_tm + q * 16;
                                    const float *k3 = kernel3_tm + q * 16;

                                    for (int n = 0; n < 16; n++)
                                    {
                                        sum0[n] += r0[n] * k0[n];
                                        k0 += 16;
                                        sum0[n] += r1[n] * k0[n];
                                        k0 += 16;
                                        sum0[n] += r2[n] * k0[n];
                                        k0 += 16;
                                        sum0[n] += r3[n] * k0[n];
                                        k0 -= 16 * 3;

                                        sum1[n] += r0[n] * k1[n];
                                        k1 += 16;
                                        sum1[n] += r1[n] * k1[n];
                                        k1 += 16;
                                        sum1[n] += r2[n] * k1[n];
                                        k1 += 16;
                                        sum1[n] += r3[n] * k1[n];
                                        k1 -= 16 * 3;

                                        sum2[n] += r0[n] * k2[n];
                                        k2 += 16;
                                        sum2[n] += r1[n] * k2[n];
                                        k2 += 16;
                                        sum2[n] += r2[n] * k2[n];
                                        k2 += 16;
                                        sum2[n] += r3[n] * k2[n];
                                        k2 -= 16 * 3;

                                        sum3[n] += r0[n] * k3[n];
                                        k3 += 16;
                                        sum3[n] += r1[n] * k3[n];
                                        k3 += 16;
                                        sum3[n] += r2[n] * k3[n];
                                        k3 += 16;
                                        sum3[n] += r3[n] * k3[n];
                                        k3 -= 16 * 3;
                                    }
                                }

                                for (; q < inch; q++)
                                {
                                    const float *r0 = bottom_blob_tm_data + q * tiles * 16 + i * 16;

                                    const float *k0 = kernel0_tm + q * 16;
                                    const float *k1 = kernel1_tm + q * 16;
                                    const float *k2 = kernel2_tm + q * 16;
                                    const float *k3 = kernel3_tm + q * 16;

                                    for (int n = 0; n < 16; n++)
                                    {
                                        sum0[n] += r0[n] * k0[n];
                                        sum1[n] += r0[n] * k1[n];
                                        sum2[n] += r0[n] * k2[n];
                                        sum3[n] += r0[n] * k3[n];
                                    }
                                }

                                for (int n = 0; n < 16; n++)
                                {
                                    output0_tm[n] = sum0[n];
                                    output1_tm[n] = sum1[n];
                                    output2_tm[n] = sum2[n];
                                    output3_tm[n] = sum3[n];
                                }
#endif
                            }
                        }
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                        for (int p = remain_ou_start; p < ou; p++)
                        {
                            float *out0_tm = top_blob_tm_data + p * 16 * tiles;
                            const float *kernel0_tm = kernel_tm_data + p * 16 * inch;
                            for (int i = 0; i < tiles; i++)
                            {
                                float *output0_tm = out0_tm + i * 16;
                                float sum0[16] = {0.0f};

                                int q = 0;
                                for (; q + 3 < inch; q += 4)
                                {
                                    const float *r0 = bottom_blob_tm_data + q * 16 * tiles + i * 16;
                                    const float *r1 = bottom_blob_tm_data + (q + 1) * 16 * tiles + i * 16;
                                    const float *r2 = bottom_blob_tm_data + (q + 2) * 16 * tiles + i * 16;
                                    const float *r3 = bottom_blob_tm_data + (q + 3) * 16 * tiles + i * 16;

                                    const float *k0 = kernel0_tm + q * 16;
                                    const float *k1 = kernel0_tm + (q + 1) * 16;
                                    const float *k2 = kernel0_tm + (q + 2) * 16;
                                    const float *k3 = kernel0_tm + (q + 3) * 16;

                                    for (int n = 0; n < 16; n++)
                                    {
                                        sum0[n] += r0[n] * k0[n];
                                        sum0[n] += r1[n] * k1[n];
                                        sum0[n] += r2[n] * k2[n];
                                        sum0[n] += r3[n] * k3[n];
                                    }
                                }
                                for (; q < inch; q++)
                                {
                                    const float *r0 = bottom_blob_tm_data + q * 16 * tiles + i * 16;
                                    const float *k0 = kernel0_tm + q * 16;
                                    for (int n = 0; n < 16; n++)
                                        sum0[n] += r0[n] * k0[n];
                                }
                                for (int n = 0; n < 16; n++)
                                    output0_tm[n] = sum0[n];
                            }
                        }
                    }
                    // END dot
                    // BEGIN transform output
                    int w_tm = outw >> 1 << 2;
                    int h_tm = outh >> 1 << 2;

                    int block_col = h_tm >> 2; // may be the block num in Feathercnn
                    int block_row = w_tm >> 2;

                    float *top_blob_tm_data = top_blob_tm->mutable_cpu_data();
                    const int tiles = block_col * block_row;
                    {

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                        for (int p = 0; p < ou; p++)
                        {
                            float *out_tm = top_blob_tm_data + p * 16 * tiles;
                            float *out = top_blob_bordered_data + n * ou * outh * outw + p * outh * outw;
                            const float bias0 = bias ? bias[p] : 0.f;

                            for (int j = 0; j < block_col; j++)
                            {
                                float *outRow0 = out + 2 * j * outw;
                                float *outRow1 = out + (2 * j + 1) * outw;

                                for (int i = 0; i < block_row; i++)
                                {
                                    float *out_tile = out_tm + (j * block_row + i) * 16;
                                    float s0[4], s1[4], s2[4], s3[4], d0[2], d1[2], d2[2], d3[2];
                                    float w0[4], w1[4], o0[2], o1[2];
                                    for (int n = 0; n < 4; n++)
                                    {
                                        s0[n] = out_tile[n];
                                        s1[n] = out_tile[n + 4];
                                        s2[n] = out_tile[n + 8];
                                        s3[n] = out_tile[n + 12];
                                    }
                                    // w = A_T * W
                                    for (int n = 0; n < 4; n++)
                                    {
                                        w0[n] = s0[n] + s1[n] + s2[n];
                                        w1[n] = s1[n] - s2[n] + s3[n];
                                    }
                                    // save to top blob tm
                                    outRow0[0] = w0[0] + w0[1] + w0[2] + bias0;
                                    outRow0[1] = w1[0] + w1[1] + w1[2] + bias0;
                                    outRow1[0] = w0[1] - w0[2] + w0[3] + bias0;
                                    outRow1[1] = w1[1] - w1[2] + w1[3] + bias0;

                                    outRow0 += 2;
                                    outRow1 += 2;
                                }
                            }
                        }
                    }
                }
                //top_blob = top_blob_bordered_;
                cut_border_cpu<float>(top_blob_bordered_, top_blob, 0, top_blob_bordered_->height() - top_blob->height(), 0, top_blob_bordered_->width() - top_blob->width());
            }
        }

        template <typename Dtype>
        void operation_convolution<Dtype>::forward_im2col_tr_kernel()
        {
            const int input_channel_ = this->weight_data_size_ * this->group_ / this->output_channel_ / this->kernel_size_h_ / this->kernel_size_w_;
            int inch = input_channel_ / this->group_;
            int outch = this->output_channel_ / this->group_;
            int kernel_size = this->kernel_size_h_ * this->kernel_size_w_;

            const float *kernel_base = this->weights_f32_[0]->cpu_data();
            // kernel memory packed 8 x 8
            int kernel_tm_channel = outch / 8 + (outch % 8) / 4 + outch % 4;
            int nn_outch = 0;
            int remain_outch_start = 0;
            nn_outch = outch >> 3;
            remain_outch_start = nn_outch << 3;

            for (int g = 0; g < this->group_; g++)
            {
                kernel_tm_gemm_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(std::vector<int>{1, kernel_tm_channel, inch, 8 * kernel_size}, this->params_.device_, memory::NCHW, nullptr)));
                float *kernel_tm = kernel_tm_gemm_[g]->mutable_cpu_data();
                int kernel_tm_cstep = kernel_tm_gemm_[g]->width() * kernel_tm_gemm_[g]->height();
                const float *kernel = kernel_base + this->weight_data_size_ * g / this->group_;
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                for (int pp = 0; pp < nn_outch; pp++)
                {
                    int p = pp * 8;

                    const float *k0 = kernel + (p + 0) * inch * kernel_size;
                    const float *k1 = kernel + (p + 1) * inch * kernel_size;
                    const float *k2 = kernel + (p + 2) * inch * kernel_size;
                    const float *k3 = kernel + (p + 3) * inch * kernel_size;
                    const float *k4 = kernel + (p + 4) * inch * kernel_size;
                    const float *k5 = kernel + (p + 5) * inch * kernel_size;
                    const float *k6 = kernel + (p + 6) * inch * kernel_size;
                    const float *k7 = kernel + (p + 7) * inch * kernel_size;

                    float *ktmp = kernel_tm + (p / 8) * kernel_tm_cstep;

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

                        k0 += 1;
                        k1 += 1;
                        k2 += 1;
                        k3 += 1;
                        k4 += 1;
                        k5 += 1;
                        k6 += 1;
                        k7 += 1;
                    }
                }

                nn_outch = (outch - remain_outch_start) >> 2;
                for (int pp = 0; pp < nn_outch; pp++)
                {
                    int p = remain_outch_start + pp * 4;
                    const float *k0 = kernel + (p + 0) * inch * kernel_size;
                    const float *k1 = kernel + (p + 1) * inch * kernel_size;
                    const float *k2 = kernel + (p + 2) * inch * kernel_size;
                    const float *k3 = kernel + (p + 3) * inch * kernel_size;

                    float *ktmp = kernel_tm + (p / 8 + (p % 8) / 4) * kernel_tm_cstep;
                    for (int q = 0; q < inch * kernel_size; q++)
                    {
                        ktmp[0] = k0[0];
                        ktmp[1] = k1[0];
                        ktmp[2] = k2[0];
                        ktmp[3] = k3[0];
                        ktmp += 4;

                        k0 += 1;
                        k1 += 1;
                        k2 += 1;
                        k3 += 1;
                    }
                }
                remain_outch_start += nn_outch << 2;
                for (int p = remain_outch_start; p < outch; p++)
                {
                    const float *k0 = kernel + (p + 0) * inch * kernel_size;
                    float *ktmp = kernel_tm + (p / 8 + (p % 8) / 4 + p % 4) * kernel_tm_cstep;
                    for (int q = 0; q < inch * kernel_size; q++)
                    {
                        ktmp[0] = k0[0];
                        ktmp++;
                        k0++;
                    }
                }
            }
        }

        template <typename Dtype>
        void operation_convolution<Dtype>::forward_im2col(const std::shared_ptr<memory::tensor<float>> &bottom_blob,
                                                          std::shared_ptr<memory::tensor<float>> &top_blob)
        {
            int num = bottom_blob->num();
            int w = bottom_blob->width();
            int h = bottom_blob->height();
            int inch = bottom_blob->channels() / this->group_;
            int bottom_cstep = w * h;
            int outw = top_blob->width();
            int outh = top_blob->height();
            int outch = top_blob->channels() / this->group_;

            const float *bias_0 = nullptr;
            if (this->bias_term_)
                bias_0 = this->weights_f32_[1]->cpu_data();

            int top_cstep = top_blob->width() * top_blob->height();
            int out_size = top_cstep;
            int kernel_size = this->kernel_size_w_ * this->kernel_size_h_;
            int N = outw * outh;        // outsize or out stride
            int L = kernel_size * inch; // ksize * inch

            int stride = kernel_size * outw * outh;

            for (int n = 0; n < num; n++)
            {
                float *top_data_base = top_blob->mutable_cpu_data() + n * top_blob->count(1, 4);
                const float *bottom_data_base = bottom_blob->cpu_data() + n * bottom_blob->count(1, 4);

                for (int g = 0; g < this->group_; g++)
                {
                    const float *kernel_tm_data = kernel_tm_gemm_[g]->cpu_data();

                    int kernel_tm_cstep = kernel_tm_gemm_[g]->width() * kernel_tm_gemm_[g]->height();
                    const float *bottom_data = bottom_data_base + g * bottom_cstep * inch;
                    const float *bias = bias_0 + g * outch;
                    float *top_data = top_data_base + out_size * outch * g;
                    {
                        // im2col
                        bottom_im2col_.reset(new memory::tensor<float>(std::vector<int>{1, 1, kernel_size * inch, out_size}, this->params_.device_, memory::NCHW, bottom_blob->allocator()));
                        float *ret = bottom_im2col_->mutable_cpu_data();
                        const float *bottom_im2col_data = bottom_im2col_->cpu_data();

                        // bottom_im2col_ memory packed 8 x 8
                        bottom_tm_.reset(new memory::tensor<float>(std::vector<int>{1, out_size / 8 + out_size % 8, inch, 8 * kernel_size}, this->params_.device_, memory::NCHW, bottom_blob->allocator()));
                        float *bottom_tm_data = bottom_tm_->mutable_cpu_data();
                        int bottom_tm_cstep = bottom_tm_->width() * bottom_tm_->height();
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                        for (int p = 0; p < inch; p++)
                        {
                            const float *input = bottom_data + (p)*bottom_cstep;
                            int retID = stride * p;
                            for (int u = 0; u < this->kernel_size_h_; u++)
                            {
                                for (int v = 0; v < this->kernel_size_w_; v++)
                                {
                                    for (int i = 0; i < outh; i++)
                                    {
                                        for (int j = 0; j < outw; j++)
                                        {
                                            int row = u + i * this->stride_h_;
                                            int col = v + j * this->stride_w_;
                                            int index = row * w + col;
                                            ret[retID] = input[index];
                                            retID++;
                                        }
                                    }
                                }
                            }
                        }
                        int nn_size = out_size >> 3;
                        int remain_size_start = nn_size << 3;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                        for (int ii = 0; ii < nn_size; ii++)
                        {
                            int i = ii * 8;

                            const float *img0 = bottom_im2col_data;
                            img0 += i;

                            float *tmpptr = bottom_tm_data + (i / 8) * bottom_tm_cstep;

                            for (int q = 0; q < inch * kernel_size; q++)
                            {
#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX //AVX
                                _mm256_storeu_ps(tmpptr, _mm256_loadu_ps(img0));
#else
                                tmpptr[0] = img0[0];
                                tmpptr[1] = img0[1];
                                tmpptr[2] = img0[2];
                                tmpptr[3] = img0[3];
                                tmpptr[4] = img0[4];
                                tmpptr[5] = img0[5];
                                tmpptr[6] = img0[6];
                                tmpptr[7] = img0[7];
#endif
                                tmpptr += 8;
                                img0 += out_size;
                            }
                        }

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                        for (int i = remain_size_start; i < out_size; i++)
                        {
                            const float *img0 = bottom_im2col_data;
                            img0 += i;
                            float *tmpptr = bottom_tm_data + (i / 8 + i % 8) * bottom_tm_cstep;
                            for (int q = 0; q < inch * kernel_size; q++)
                            {
                                tmpptr[0] = img0[0];
                                tmpptr += 1;
                                img0 += out_size;
                            }
                        }
                        int nn_outch = 0;
                        int remain_outch_start = 0;

                        nn_outch = outch >> 3;
                        remain_outch_start = nn_outch << 3;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                        for (int pp = 0; pp < nn_outch; pp++)
                        {
                            int i = pp * 8;

                            float *output0 = top_data + (i)*top_cstep;
                            float *output1 = top_data + (i + 1) * top_cstep;
                            float *output2 = top_data + (i + 2) * top_cstep;
                            float *output3 = top_data + (i + 3) * top_cstep;
                            float *output4 = top_data + (i + 4) * top_cstep;
                            float *output5 = top_data + (i + 5) * top_cstep;
                            float *output6 = top_data + (i + 6) * top_cstep;
                            float *output7 = top_data + (i + 7) * top_cstep;

                            const float zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                            const float *biasptr = bias ? bias + i : zeros;

                            int j = 0;
                            for (; j + 7 < N; j = j + 8)
                            {
                                const float *vb = bottom_tm_data + (j / 8) * bottom_tm_cstep;
                                const float *va = kernel_tm_data + (i / 8) * kernel_tm_cstep;

#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX
                                __m256 _sum0 = _mm256_broadcast_ss(biasptr);
                                __m256 _sum1 = _mm256_broadcast_ss(biasptr + 1);
                                __m256 _sum2 = _mm256_broadcast_ss(biasptr + 2);
                                __m256 _sum3 = _mm256_broadcast_ss(biasptr + 3);
                                __m256 _sum4 = _mm256_broadcast_ss(biasptr + 4);
                                __m256 _sum5 = _mm256_broadcast_ss(biasptr + 5);
                                __m256 _sum6 = _mm256_broadcast_ss(biasptr + 6);
                                __m256 _sum7 = _mm256_broadcast_ss(biasptr + 7);

                                int k = 0;
                                {
#include "operation_cxx/op_im2col_pack8_sse1.cxx"
                                }
                                for (; k < L; k++)
                                {
                                    // k0
                                    __m256 _va0 = _mm256_broadcast_ss(va);
                                    __m256 _va1 = _mm256_broadcast_ss(va + 1);
                                    __m256 _va2 = _mm256_broadcast_ss(va + 2);
                                    __m256 _va3 = _mm256_broadcast_ss(va + 3);
                                    __m256 _va4 = _mm256_broadcast_ss(va + 4);
                                    __m256 _va5 = _mm256_broadcast_ss(va + 5);
                                    __m256 _va6 = _mm256_broadcast_ss(va + 6);
                                    __m256 _va7 = _mm256_broadcast_ss(va + 7);
                                    __m256 _vb0 = _mm256_loadu_ps(vb);
                                    _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0); // sum0 = (a00-a07) * k00
                                    _sum1 = _mm256_fmadd_ps(_vb0, _va1, _sum1); // sum1 = (a00-a07) * k10
                                    _sum2 = _mm256_fmadd_ps(_vb0, _va2, _sum2); // sum2 = (a00-a07) * k20
                                    _sum3 = _mm256_fmadd_ps(_vb0, _va3, _sum3); // sum3 = (a00-a07) * k30
                                    _sum4 = _mm256_fmadd_ps(_vb0, _va4, _sum4); // sum4 = (a00-a07) * k40
                                    _sum5 = _mm256_fmadd_ps(_vb0, _va5, _sum5); // sum5 = (a00-a07) * k50
                                    _sum6 = _mm256_fmadd_ps(_vb0, _va6, _sum6); // sum6 = (a00-a07) * k60
                                    _sum7 = _mm256_fmadd_ps(_vb0, _va7, _sum7); // sum7 = (a00-a07) * k70

                                    va += 8;
                                    vb += 8;
                                }

                                _mm256_storeu_ps(output0, _sum0);
                                _mm256_storeu_ps(output1, _sum1);
                                _mm256_storeu_ps(output2, _sum2);
                                _mm256_storeu_ps(output3, _sum3);
                                _mm256_storeu_ps(output4, _sum4);
                                _mm256_storeu_ps(output5, _sum5);
                                _mm256_storeu_ps(output6, _sum6);
                                _mm256_storeu_ps(output7, _sum7);
#else
                                float sum0[8] = {0};
                                float sum1[8] = {0};
                                float sum2[8] = {0};
                                float sum3[8] = {0};
                                float sum4[8] = {0};
                                float sum5[8] = {0};
                                float sum6[8] = {0};
                                float sum7[8] = {0};

#include "operation_cxx/op_im2col_native1.cxx"

                                for (int n = 0; n < 8; n++)
                                {
                                    output0[n] = sum0[n] + biasptr[0];
                                    output1[n] = sum1[n] + biasptr[1];
                                    output2[n] = sum2[n] + biasptr[2];
                                    output3[n] = sum3[n] + biasptr[3];
                                    output4[n] = sum4[n] + biasptr[4];
                                    output5[n] = sum5[n] + biasptr[5];
                                    output6[n] = sum6[n] + biasptr[6];
                                    output7[n] = sum7[n] + biasptr[7];
                                }
#endif // __AVX__
                                output0 += 8;
                                output1 += 8;
                                output2 += 8;
                                output3 += 8;
                                output4 += 8;
                                output5 += 8;
                                output6 += 8;
                                output7 += 8;
                            }

                            for (; j < N; j++)
                            {
                                const float *vb = bottom_tm_data + (j / 8 + j % 8) * bottom_tm_cstep;
                                const float *va = kernel_tm_data + (i / 8) * kernel_tm_cstep;
#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX //AVX
                                __m256 _sum0_7 = _mm256_loadu_ps(biasptr);
                                __m256 _sum0 = _mm256_set1_ps(0.0);
                                __m256 _sum1 = _mm256_set1_ps(0.0);
                                __m256 _sum2 = _mm256_set1_ps(0.0);
                                __m256 _sum3 = _mm256_set1_ps(0.0);

                                int k = 0;
                                for (; k + 3 < L; k = k + 4)
                                {
                                    __m256 _vb0 = _mm256_broadcast_ss(vb);
                                    __m256 _vb1 = _mm256_broadcast_ss(vb + 1);
                                    __m256 _vb2 = _mm256_broadcast_ss(vb + 2);
                                    __m256 _vb3 = _mm256_broadcast_ss(vb + 3);
                                    __m256 _va0 = _mm256_loadu_ps(va);
                                    __m256 _va1 = _mm256_loadu_ps(va + 8);
                                    __m256 _va2 = _mm256_loadu_ps(va + 16);
                                    __m256 _va3 = _mm256_loadu_ps(va + 24);

                                    _sum0 = _mm256_fmadd_ps(_va0, _vb0, _sum0); // sum0 += (k00-k70) * a00
                                    _sum1 = _mm256_fmadd_ps(_va1, _vb1, _sum1); // sum1 += (k01-k71) * a10
                                    _sum2 = _mm256_fmadd_ps(_va2, _vb2, _sum2); // sum2 += (k02-k72) * a20
                                    _sum3 = _mm256_fmadd_ps(_va3, _vb3, _sum3); // sum3 += (k03-k73) * a30

                                    va += 32;
                                    vb += 4;
                                }

                                _sum0 = _mm256_add_ps(_sum0, _sum1);
                                _sum2 = _mm256_add_ps(_sum2, _sum3);
                                _sum0_7 = _mm256_add_ps(_sum0_7, _sum0);
                                _sum0_7 = _mm256_add_ps(_sum0_7, _sum2);

                                for (; k < L; k++)
                                {
                                    __m256 _vb0 = _mm256_broadcast_ss(vb);
                                    __m256 _va = _mm256_loadu_ps(va);

                                    _sum0_7 = _mm256_fmadd_ps(_va, _vb0, _sum0_7); // sum0 += (k00-k70) * a00

                                    va += 8;
                                    vb += 1;
                                }

                                float output_sum0_7[8] = {0.f};
                                _mm256_storeu_ps(output_sum0_7, _sum0_7);

                                output0[0] = output_sum0_7[0];
                                output1[0] = output_sum0_7[1];
                                output2[0] = output_sum0_7[2];
                                output3[0] = output_sum0_7[3];
                                output4[0] = output_sum0_7[4];
                                output5[0] = output_sum0_7[5];
                                output6[0] = output_sum0_7[6];
                                output7[0] = output_sum0_7[7];
#else
                                float sum0 = biasptr[0];
                                float sum1 = biasptr[1];
                                float sum2 = biasptr[2];
                                float sum3 = biasptr[3];
                                float sum4 = biasptr[4];
                                float sum5 = biasptr[5];
                                float sum6 = biasptr[6];
                                float sum7 = biasptr[7];

                                for (int k = 0; k < L; k++)
                                {
                                    sum0 += va[0] * vb[0];
                                    sum1 += va[1] * vb[0];
                                    sum2 += va[2] * vb[0];
                                    sum3 += va[3] * vb[0];
                                    sum4 += va[4] * vb[0];
                                    sum5 += va[5] * vb[0];
                                    sum6 += va[6] * vb[0];
                                    sum7 += va[7] * vb[0];

                                    va += 8;
                                    vb += 1;
                                }

                                output0[0] = sum0;
                                output1[0] = sum1;
                                output2[0] = sum2;
                                output3[0] = sum3;
                                output4[0] = sum4;
                                output5[0] = sum5;
                                output6[0] = sum6;
                                output7[0] = sum7;
#endif // __AVX__

                                output0++;
                                output1++;
                                output2++;
                                output3++;
                                output4++;
                                output5++;
                                output6++;
                                output7++;
                            }
                        }
                        nn_outch = (outch - remain_outch_start) >> 2;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                        for (int pp = 0; pp < nn_outch; pp++)
                        {
                            int i = remain_outch_start + pp * 4;

                            float *output0 = top_data + (i)*top_cstep;
                            float *output1 = top_data + (i + 1) * top_cstep;
                            float *output2 = top_data + (i + 2) * top_cstep;
                            float *output3 = top_data + (i + 3) * top_cstep;

                            const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
                            const float *biasptr = bias ? bias + i : zeros;

                            int j = 0;
                            for (; j + 7 < N; j = j + 8)
                            {
                                const float *vb = bottom_tm_data + (j / 8) * bottom_tm_cstep;
                                const float *va = kernel_tm_data + (i / 8 + (i % 8) / 4) * kernel_tm_cstep;
#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX //AVX
                                __m256 _sum0 = _mm256_broadcast_ss(biasptr);
                                __m256 _sum1 = _mm256_broadcast_ss(biasptr + 1);
                                __m256 _sum2 = _mm256_broadcast_ss(biasptr + 2);
                                __m256 _sum3 = _mm256_broadcast_ss(biasptr + 3);

                                int k = 0;
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
                                    _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0); // sum0 = (a00-a07) * k00
                                    _sum1 = _mm256_fmadd_ps(_vb0, _va1, _sum1); // sum1 = (a00-a07) * k10
                                    _sum2 = _mm256_fmadd_ps(_vb0, _va2, _sum2); // sum2 = (a00-a07) * k20
                                    _sum3 = _mm256_fmadd_ps(_vb0, _va3, _sum3); // sum3 = (a00-a07) * k30

                                    va += 4;

                                    // k1
                                    _va0 = _mm256_broadcast_ss(va);
                                    _va1 = _mm256_broadcast_ss(va + 1);
                                    _va2 = _mm256_broadcast_ss(va + 2);
                                    _va3 = _mm256_broadcast_ss(va + 3);
                                    _sum0 = _mm256_fmadd_ps(_vb1, _va0, _sum0); // sum0 += (a10-a17) * k01
                                    _sum1 = _mm256_fmadd_ps(_vb1, _va1, _sum1); // sum1 += (a10-a17) * k11
                                    _sum2 = _mm256_fmadd_ps(_vb1, _va2, _sum2); // sum2 += (a10-a17) * k21
                                    _sum3 = _mm256_fmadd_ps(_vb1, _va3, _sum3); // sum3 += (a10-a17) * k31

                                    va += 4;

                                    // k2
                                    _va0 = _mm256_broadcast_ss(va);
                                    _va1 = _mm256_broadcast_ss(va + 1);
                                    _va2 = _mm256_broadcast_ss(va + 2);
                                    _va3 = _mm256_broadcast_ss(va + 3);
                                    _sum0 = _mm256_fmadd_ps(_vb2, _va0, _sum0); // sum0 += (a20-a27) * k02
                                    _sum1 = _mm256_fmadd_ps(_vb2, _va1, _sum1); // sum1 += (a20-a27) * k12
                                    _sum2 = _mm256_fmadd_ps(_vb2, _va2, _sum2); // sum2 += (a20-a27) * k22
                                    _sum3 = _mm256_fmadd_ps(_vb2, _va3, _sum3); // sum3 += (a20-a27) * k32

                                    va += 4;

                                    // k3
                                    _va0 = _mm256_broadcast_ss(va);
                                    _va1 = _mm256_broadcast_ss(va + 1);
                                    _va2 = _mm256_broadcast_ss(va + 2);
                                    _va3 = _mm256_broadcast_ss(va + 3);
                                    _sum0 = _mm256_fmadd_ps(_vb3, _va0, _sum0); // sum0 += (a30-a37) * k03
                                    _sum1 = _mm256_fmadd_ps(_vb3, _va1, _sum1); // sum1 += (a30-a37) * k13
                                    _sum2 = _mm256_fmadd_ps(_vb3, _va2, _sum2); // sum2 += (a30-a37) * k23
                                    _sum3 = _mm256_fmadd_ps(_vb3, _va3, _sum3); // sum3 += (a30-a37) * k33

                                    va += 4;
                                    vb += 32;
                                }

                                for (; k < L; k++)
                                {
                                    // k0
                                    __m256 _va0 = _mm256_broadcast_ss(va);
                                    __m256 _va1 = _mm256_broadcast_ss(va + 1);
                                    __m256 _va2 = _mm256_broadcast_ss(va + 2);
                                    __m256 _va3 = _mm256_broadcast_ss(va + 3);
                                    __m256 _vb0 = _mm256_loadu_ps(vb);
                                    _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0); // sum0 = (a00-a07) * k00
                                    _sum1 = _mm256_fmadd_ps(_vb0, _va1, _sum1); // sum1 = (a00-a07) * k10
                                    _sum2 = _mm256_fmadd_ps(_vb0, _va2, _sum2); // sum2 = (a00-a07) * k20
                                    _sum3 = _mm256_fmadd_ps(_vb0, _va3, _sum3); // sum3 = (a00-a07) * k30

                                    va += 4;
                                    vb += 8;
                                }

                                _mm256_storeu_ps(output0, _sum0);
                                _mm256_storeu_ps(output1, _sum1);
                                _mm256_storeu_ps(output2, _sum2);
                                _mm256_storeu_ps(output3, _sum3);
#else
                                float sum0[8] = {0};
                                float sum1[8] = {0};
                                float sum2[8] = {0};
                                float sum3[8] = {0};

                                int k = 0;
                                for (; k + 7 < L; k = k + 8)
                                {
                                    for (int n = 0; n < 8; n++)
                                    {
                                        sum0[n] += va[0] * vb[n];
                                        sum1[n] += va[1] * vb[n];
                                        sum2[n] += va[2] * vb[n];
                                        sum3[n] += va[3] * vb[n];
                                        va += 4;

                                        sum0[n] += va[0] * vb[n + 8];
                                        sum1[n] += va[1] * vb[n + 8];
                                        sum2[n] += va[2] * vb[n + 8];
                                        sum3[n] += va[3] * vb[n + 8];
                                        va += 4;

                                        sum0[n] += va[0] * vb[n + 16];
                                        sum1[n] += va[1] * vb[n + 16];
                                        sum2[n] += va[2] * vb[n + 16];
                                        sum3[n] += va[3] * vb[n + 16];
                                        va += 4;

                                        sum0[n] += va[0] * vb[n + 24];
                                        sum1[n] += va[1] * vb[n + 24];
                                        sum2[n] += va[2] * vb[n + 24];
                                        sum3[n] += va[3] * vb[n + 24];
                                        va += 4;

                                        sum0[n] += va[0] * vb[n + 32];
                                        sum1[n] += va[1] * vb[n + 32];
                                        sum2[n] += va[2] * vb[n + 32];
                                        sum3[n] += va[3] * vb[n + 32];
                                        va += 4;

                                        sum0[n] += va[0] * vb[n + 40];
                                        sum1[n] += va[1] * vb[n + 40];
                                        sum2[n] += va[2] * vb[n + 40];
                                        sum3[n] += va[3] * vb[n + 40];
                                        va += 4;

                                        sum0[n] += va[0] * vb[n + 48];
                                        sum1[n] += va[1] * vb[n + 48];
                                        sum2[n] += va[2] * vb[n + 48];
                                        sum3[n] += va[3] * vb[n + 48];
                                        va += 4;

                                        sum0[n] += va[0] * vb[n + 56];
                                        sum1[n] += va[1] * vb[n + 56];
                                        sum2[n] += va[2] * vb[n + 56];
                                        sum3[n] += va[3] * vb[n + 56];
                                        va -= 28;
                                    }

                                    va += 32;
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
                                    }

                                    va += 4;
                                    vb += 8;
                                }

                                for (int n = 0; n < 8; n++)
                                {
                                    output0[n] = sum0[n] + biasptr[0];
                                    output1[n] = sum1[n] + biasptr[1];
                                    output2[n] = sum2[n] + biasptr[2];
                                    output3[n] = sum3[n] + biasptr[3];
                                }
#endif // __AVX__
                                output0 += 8;
                                output1 += 8;
                                output2 += 8;
                                output3 += 8;
                            }

                            for (; j < N; j++)
                            {
                                float *vb = bottom_tm_data + (j / 8 + j % 8) * bottom_tm_cstep;
                                const float *va = kernel_tm_data + (i / 8 + (i % 8) / 4) * kernel_tm_cstep;
#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX //AVX
                                __m128 _sum0_3 = _mm_loadu_ps(biasptr);
                                __m128 _sum0 = _mm_set1_ps(0.0);
                                __m128 _sum1 = _mm_set1_ps(0.0);
                                __m128 _sum2 = _mm_set1_ps(0.0);
                                __m128 _sum3 = _mm_set1_ps(0.0);

                                int k = 0;
                                for (; k + 3 < L; k = k + 4)
                                {
                                    __m128 _vb0 = _mm_set1_ps(vb[0]);
                                    __m128 _vb1 = _mm_set1_ps(vb[1]);
                                    __m128 _vb2 = _mm_set1_ps(vb[2]);
                                    __m128 _vb3 = _mm_set1_ps(vb[3]);
                                    __m128 _va0 = _mm_loadu_ps(va);
                                    __m128 _va1 = _mm_loadu_ps(va + 4);
                                    __m128 _va2 = _mm_loadu_ps(va + 8);
                                    __m128 _va3 = _mm_loadu_ps(va + 12);

                                    _sum0 = _mm_fmadd_ps(_va0, _vb0, _sum0); // sum0 += (k00-k30) * a00
                                    _sum1 = _mm_fmadd_ps(_va1, _vb1, _sum1); // sum1 += (k01-k31) * a10
                                    _sum2 = _mm_fmadd_ps(_va2, _vb2, _sum2); // sum2 += (k02-k32) * a20
                                    _sum3 = _mm_fmadd_ps(_va3, _vb3, _sum3); // sum3 += (k03-k33) * a30

                                    va += 16;
                                    vb += 4;
                                }

                                _sum0 = _mm_add_ps(_sum0, _sum1);
                                _sum2 = _mm_add_ps(_sum2, _sum3);
                                _sum0_3 = _mm_add_ps(_sum0_3, _sum0);
                                _sum0_3 = _mm_add_ps(_sum0_3, _sum2);

                                for (; k < L; k++)
                                {
                                    __m128 _vb0 = _mm_set1_ps(vb[0]);
                                    __m128 _va = _mm_loadu_ps(va);

                                    _sum0_3 = _mm_fmadd_ps(_va, _vb0, _sum0_3); // sum0 += (k00-k30) * a00

                                    va += 4;
                                    vb += 1;
                                }

                                float output_sum0_3[4] = {0.f};
                                _mm_storeu_ps(output_sum0_3, _sum0_3);
                                output0[0] = output_sum0_3[0];
                                output1[0] = output_sum0_3[1];
                                output2[0] = output_sum0_3[2];
                                output3[0] = output_sum0_3[3];
#else
                                float sum0 = biasptr[0];
                                float sum1 = biasptr[1];
                                float sum2 = biasptr[2];
                                float sum3 = biasptr[3];

                                for (int k = 0; k < L; k++)
                                {
                                    sum0 += va[0] * vb[0];
                                    sum1 += va[1] * vb[0];
                                    sum2 += va[2] * vb[0];
                                    sum3 += va[3] * vb[0];

                                    va += 4;
                                    vb += 1;
                                }

                                output0[0] = sum0;
                                output1[0] = sum1;
                                output2[0] = sum2;
                                output3[0] = sum3;
#endif
                                output0++;
                                output1++;
                                output2++;
                                output3++;
                            }
                        }

                        remain_outch_start += nn_outch << 2;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                        for (int i = remain_outch_start; i < outch; i++)
                        {
                            float *output = top_data + (i)*top_cstep;

                            const float bias0 = bias ? bias[i] : 0.f;
                            int j = 0;
                            for (; j + 7 < N; j = j + 8)
                            {
                                const float *vb = bottom_tm_data + (j / 8) * bottom_tm_cstep;
                                const float *va = kernel_tm_data + (i / 8 + (i % 8) / 4 + i % 4) * kernel_tm_cstep;
#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX //AVX
                                __m256 _sum0 = _mm256_broadcast_ss(&bias0);

                                int k = 0;
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

                                    _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0); // sum0 = (a00-a07) * k00
                                    _sum0 = _mm256_fmadd_ps(_vb1, _va1, _sum0); // sum0 += (a10-a17) * k01
                                    _sum0 = _mm256_fmadd_ps(_vb2, _va2, _sum0); // sum0 += (a20-a27) * k02
                                    _sum0 = _mm256_fmadd_ps(_vb3, _va3, _sum0); // sum0 += (a30-a37) * k03

                                    va += 4;
                                    vb += 32;
                                }

                                for (; k < L; k++)
                                {
                                    // k0
                                    __m256 _va0 = _mm256_broadcast_ss(va);
                                    __m256 _vb0 = _mm256_loadu_ps(vb);

                                    _sum0 = _mm256_fmadd_ps(_vb0, _va0, _sum0); // sum0 = (a00-a07) * k00

                                    va += 1;
                                    vb += 8;
                                }

                                _mm256_storeu_ps(output, _sum0);
#else
                                float sum[8] = {0};

                                int k = 0;
                                for (; k + 7 < L; k = k + 8)
                                {
                                    for (int n = 0; n < 8; n++)
                                    {
                                        sum[n] += va[0] * vb[n];
                                        sum[n] += va[1] * vb[n + 8];
                                        sum[n] += va[2] * vb[n + 16];
                                        sum[n] += va[3] * vb[n + 24];
                                        sum[n] += va[4] * vb[n + 32];
                                        sum[n] += va[5] * vb[n + 40];
                                        sum[n] += va[6] * vb[n + 48];
                                        sum[n] += va[7] * vb[n + 56];
                                    }

                                    va += 8;
                                    vb += 64;
                                }

                                for (; k < L; k++)
                                {
                                    for (int n = 0; n < 8; n++)
                                    {
                                        sum[n] += va[0] * vb[n];
                                    }

                                    va += 1;
                                    vb += 8;
                                }

                                for (int n = 0; n < 8; n++)
                                {
                                    output[n] = sum[n] + bias0;
                                }
#endif // __AVX__
                                output += 8;
                            }
                            for (; j < N; j++)
                            {
                                const float *vb = bottom_tm_data + (j / 8 + j % 8) * bottom_tm_cstep;
                                const float *va = kernel_tm_data + (i / 8 + (i % 8) / 4 + i % 4) * kernel_tm_cstep;

                                int k = 0;
#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX //AVX
                                __m128 _sum0 = _mm_set1_ps(0.f);

                                for (; k + 3 < L; k += 4)
                                {
                                    __m128 _p0 = _mm_loadu_ps(vb);
                                    vb += 4;
                                    __m128 _k0 = _mm_loadu_ps(va);
                                    va += 4;

                                    _sum0 = _mm_fmadd_ps(_p0, _k0, _sum0);
                                }
                                float output_sum0[4] = {0.f};
                                _mm_storeu_ps(output_sum0, _sum0);

                                float sum0 = bias0 + output_sum0[0] + output_sum0[1] + output_sum0[2] + output_sum0[3];
#else
                                float sum0 = bias0;
#endif // __AVX__
                                for (; k < L; k++)
                                {
                                    sum0 += va[0] * vb[0];
                                    va += 1;
                                    vb += 1;
                                }
                                output[0] = sum0;

                                output++;
                            }
                        }
                    }
                }
            }
        }

        template <typename Dtype>
        void operation_convolution<Dtype>::conv3x3s1_winograd23_tr_kernel_int8()
        {
            const int input_channel_ = this->weight_data_size_ * this->group_ / this->output_channel_ / this->kernel_size_h_ / this->kernel_size_w_;
            int inch = input_channel_ / this->group_;
            int outch = this->output_channel_ / this->group_;
            const signed char *kernel_data_base = this->weights_i8_[0]->cpu_data();

            // G
            const short ktm[4][3] = {
                {2, 0, 0},
                {1, 1, 1},
                {1, -1, 1},
                {0, 0, 2}};

            for (int g = 0; g < this->group_; g++)
            {
                kernel_tm_int8_.push_back(std::shared_ptr<memory::tensor<short>>(new memory::tensor<short>(std::vector<int>{1, outch, inch, 4 * 4}, this->params_.device_, memory::NCHW, nullptr)));
                short *kernel_tm_data = kernel_tm_int8_[g]->mutable_cpu_data();
                int kernel_tm_w = kernel_tm_int8_[g]->width();
                int kernel_tm_h = kernel_tm_int8_[g]->height();
                int kernel_tm_cstep = kernel_tm_w * kernel_tm_h;
                const signed char *kernel_data = kernel_data_base + this->weight_data_size_ * g / this->group_;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                for (int p = 0; p < outch; p++)
                {
                    for (int q = 0; q < inch; q++)
                    {
                        const signed char *kernel0 = kernel_data + p * inch * 9 + q * 9;
                        short *kernel_tm0 = kernel_tm_data + p * kernel_tm_cstep + q * kernel_tm_w;
                        const signed char *k0 = kernel0;
                        const signed char *k1 = kernel0 + 3;
                        const signed char *k2 = kernel0 + 6;

                        short tmp[4][3];
                        for (int i = 0; i < 4; i++)
                        {
                            tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                            tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                            tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
                        }
                        for (int j = 0; j < 4; j++)
                        {
                            short *tmpp = &tmp[j][0];
                            for (int i = 0; i < 4; i++)
                                kernel_tm0[j * 4 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                        }
                    }
                }
            }
        }

        template <typename Dtype>
        void operation_convolution<Dtype>::conv3x3s1_winograd23_int8(const std::shared_ptr<memory::tensor<signed char>> &bottom_blob,
                                                                     std::shared_ptr<memory::tensor<int>> &top_blob)
        {
            int num = bottom_blob->num();
            int w = bottom_blob->width();
            int h = bottom_blob->height();
            int inch = bottom_blob->channels() / this->group_;
            top_blob.reset(new memory::tensor<int>(std::vector<int>{num, this->output_channel_, h - 2, w - 2}, -1, memory::NCHW, nullptr));

            int outw = top_blob->width();
            int outh = top_blob->height();
            int ou = top_blob->channels() / this->group_;

            // pad to 2n+2, winograd F(2,3)
            outw = (outw + 1) >> 1 << 1;
            outh = (outh + 1) >> 1 << 1;
            w = outw + 2;
            h = outh + 2;
            std::shared_ptr<memory::tensor<signed char>> bottom_blob_int8_bordered = bottom_blob;
            make_border<signed char>(bottom_blob, bottom_blob_int8_bordered, 0, h - bottom_blob->height(), 0, w - bottom_blob->width(), border_constant, 0);

            signed char *bottom_blob_bordered_data = bottom_blob_int8_bordered->mutable_cpu_data();
            int bordered_h = bottom_blob_int8_bordered->height();
            int bordered_w = bottom_blob_int8_bordered->width();

            //const float* bias_0 = nullptr;
            //if (this->bias_term_)
            //{
            //	bias_0 = this->weights_f32_[1]->cpu_data();
            //}

            std::shared_ptr<memory::tensor<int>> top_blob_int8_bordered;
            top_blob_int8_bordered.reset(new memory::tensor<int>(std::vector<int>{num, this->output_channel_, outh, outw}));
            int *top_data_base = top_blob_int8_bordered->mutable_cpu_data();

            for (int n = 0; n < num; n++)
            {
                signed char *bottom_data_base = bottom_blob_bordered_data + n * inch * bordered_h * bordered_w * this->group_;
                for (int g = 0; g < this->group_; g++)
                {
                    const short *kernel_tm_data = kernel_tm_int8_[g]->cpu_data();
                    const signed char *bottom_blob_bordered_data_n = bottom_data_base + g * inch * bordered_h * bordered_w;
                    //const float* bias = bias_0 + g * ou;
                    int *top_blob_bordered_data = top_data_base + g * ou * outh * outw;
                    {
                        int w_tm = outw >> 1 << 2;
                        int h_tm = outh >> 1 << 2;
                        int block_col = h_tm >> 2;
                        int block_row = w_tm >> 2;

                        const int tiles = block_col * block_row;
                        bottom_blob_int8_tm_.reset(new memory::tensor<short>(std::vector<int>{1, inch, tiles, 16}));
                        short *bottom_blob_tm_data = bottom_blob_int8_tm_->mutable_cpu_data();

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                        for (int q = 0; q < inch; q++)
                        {
                            const signed char *img = bottom_blob_bordered_data_n + q * bordered_h * bordered_w;
                            short *out_tm0 = bottom_blob_tm_data + q * 16 * tiles;

                            for (int j = 0; j < block_col; j++)
                            {
                                const signed char *r0 = img + w * j * 2;
                                const signed char *r1 = r0 + w;
                                const signed char *r2 = r1 + w;
                                const signed char *r3 = r2 + w;

                                for (int i = 0; i < block_row; i++)
                                {

                                    short d0[4], d1[4], d2[4], d3[4];
                                    short w0[4], w1[4], w2[4], w3[4];
                                    short t0[4], t1[4], t2[4], t3[4];
                                    // load
                                    {
#include "operation_cxx/op_wino23_tr_kernel.cxx"
                                    } r0 += 2;
                                    r1 += 2;
                                    r2 += 2;
                                    r3 += 2;

                                    out_tm0 += 16;
                                }
                            }
                        }
                    }
                    // BEGIN dot
                    std::shared_ptr<memory::tensor<int>> top_blob_tm;
                    {
                        int w_tm = outw >> 1 << 2;
                        int h_tm = outh >> 1 << 2;

                        int block_col = h_tm >> 2; // may be the block num in Feathercnn
                        int block_row = w_tm >> 2;
                        const int tiles = block_col * block_row;
                        top_blob_tm.reset(new memory::tensor<int>(std::vector<int>{1, ou, tiles, 16}));
                        int *top_blob_tm_data = top_blob_tm->mutable_cpu_data();
                        short *bottom_blob_tm_data = bottom_blob_int8_tm_->mutable_cpu_data();

                        int nn_ou = ou >> 2;
                        int remain_ou_start = nn_ou << 2;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                        for (int pp = 0; pp < nn_ou; pp++)
                        {
                            int p = pp << 2;

                            int *out0_tm = top_blob_tm_data + p * 16 * tiles;
                            int *out1_tm = top_blob_tm_data + (p + 1) * 16 * tiles;
                            int *out2_tm = top_blob_tm_data + (p + 2) * 16 * tiles;
                            int *out3_tm = top_blob_tm_data + (p + 3) * 16 * tiles;

                            const short *kernel0_tm = kernel_tm_data + p * inch * 16;
                            const short *kernel1_tm = kernel_tm_data + (p + 1) * inch * 16;
                            const short *kernel2_tm = kernel_tm_data + (p + 2) * inch * 16;
                            const short *kernel3_tm = kernel_tm_data + (p + 3) * inch * 16;

                            for (int i = 0; i < tiles; i++)
                            {
                                int *output0_tm = out0_tm + i * 16;
                                int *output1_tm = out1_tm + i * 16;
                                int *output2_tm = out2_tm + i * 16;
                                int *output3_tm = out3_tm + i * 16;

                                int sum0[16] = {0};
                                int sum1[16] = {0};
                                int sum2[16] = {0};
                                int sum3[16] = {0};

#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX

                                __m256i _sum0_0 = _mm256_setzero_si256();
                                __m256i _sum0_1 = _mm256_setzero_si256();
                                __m256i _sum1_0 = _mm256_setzero_si256();
                                __m256i _sum1_1 = _mm256_setzero_si256();
                                __m256i _sum2_0 = _mm256_setzero_si256();
                                __m256i _sum2_1 = _mm256_setzero_si256();
                                __m256i _sum3_0 = _mm256_setzero_si256();
                                __m256i _sum3_1 = _mm256_setzero_si256();

                                int q = 0;
                                {
#include "operation_cxx/op_wino23_int8_sse1.cxx"
                                }
                                for (; q < inch; q++)
                                {
                                    const short *r0 = bottom_blob_tm_data + q * tiles * 16 + i * 16;
                                    const short *k0 = kernel0_tm + q * 16;
                                    const short *k1 = kernel1_tm + q * 16;
                                    const short *k2 = kernel2_tm + q * 16;
                                    const short *k3 = kernel3_tm + q * 16;

                                    __m128i _r0 = _mm_loadu_si128((__m128i *)r0);
                                    __m128i _r0n = _mm_loadu_si128((__m128i *)(r0 + 8));

                                    __m128i _k0 = _mm_loadu_si128((__m128i *)k0);
                                    __m128i _k0n = _mm_loadu_si128((__m128i *)(k0 + 8));
                                    __m128i _k1 = _mm_loadu_si128((__m128i *)k1);
                                    __m128i _k1n = _mm_loadu_si128((__m128i *)(k1 + 8));
                                    __m128i _k2 = _mm_loadu_si128((__m128i *)k2);
                                    __m128i _k2n = _mm_loadu_si128((__m128i *)(k2 + 8));
                                    __m128i _k3 = _mm_loadu_si128((__m128i *)k3);
                                    __m128i _k3n = _mm_loadu_si128((__m128i *)(k3 + 8));

                                    _sum0_0 = _mm256__epi32(_r0, _k0, _sum0_0);
                                    _sum0_1 = _mm256__epi32(_r0n, _k0n, _sum0_1);
                                    _sum1_0 = _mm256__epi32(_r0, _k1, _sum1_0);
                                    _sum1_1 = _mm256__epi32(_r0n, _k1n, _sum1_1);
                                    _sum2_0 = _mm256__epi32(_r0, _k2, _sum2_0);
                                    _sum2_1 = _mm256__epi32(_r0n, _k2n, _sum2_1);
                                    _sum3_0 = _mm256__epi32(_r0, _k3, _sum3_0);
                                    _sum3_1 = _mm256__epi32(_r0n, _k3n, _sum3_1);
                                }

                                _mm256_storeu_si256((__m256i *)output0_tm, _sum0_0);
                                _mm256_storeu_si256((__m256i *)(output0_tm + 8), _sum0_1);
                                _mm256_storeu_si256((__m256i *)output1_tm, _sum1_0);
                                _mm256_storeu_si256((__m256i *)(output1_tm + 8), _sum1_1);
                                _mm256_storeu_si256((__m256i *)output2_tm, _sum2_0);
                                _mm256_storeu_si256((__m256i *)(output2_tm + 8), _sum2_1);
                                _mm256_storeu_si256((__m256i *)output3_tm, _sum3_0);
                                _mm256_storeu_si256((__m256i *)(output3_tm + 8), _sum3_1);

#else
                                int q = 0;
                                for (; q + 3 < inch; q += 4)
                                {
                                    const short *r0 = bottom_blob_tm_data + q * tiles * 16 + i * 16;
                                    const short *r1 = bottom_blob_tm_data + (q + 1) * tiles * 16 + i * 16;
                                    const short *r2 = bottom_blob_tm_data + (q + 2) * tiles * 16 + i * 16;
                                    const short *r3 = bottom_blob_tm_data + (q + 3) * tiles * 16 + i * 16;
                                    const short *k0 = kernel0_tm + q * 16;
                                    const short *k1 = kernel1_tm + q * 16;
                                    const short *k2 = kernel2_tm + q * 16;
                                    const short *k3 = kernel3_tm + q * 16;

                                    for (int n = 0; n < 16; n++)
                                    {
                                        sum0[n] += (int)r0[n] * k0[n];
                                        k0 += 16;
                                        sum0[n] += (int)r1[n] * k0[n];
                                        k0 += 16;
                                        sum0[n] += (int)r2[n] * k0[n];
                                        k0 += 16;
                                        sum0[n] += (int)r3[n] * k0[n];
                                        k0 -= 16 * 3;

                                        sum1[n] += (int)r0[n] * k1[n];
                                        k1 += 16;
                                        sum1[n] += (int)r1[n] * k1[n];
                                        k1 += 16;
                                        sum1[n] += (int)r2[n] * k1[n];
                                        k1 += 16;
                                        sum1[n] += (int)r3[n] * k1[n];
                                        k1 -= 16 * 3;

                                        sum2[n] += (int)r0[n] * k2[n];
                                        k2 += 16;
                                        sum2[n] += (int)r1[n] * k2[n];
                                        k2 += 16;
                                        sum2[n] += (int)r2[n] * k2[n];
                                        k2 += 16;
                                        sum2[n] += (int)r3[n] * k2[n];
                                        k2 -= 16 * 3;

                                        sum3[n] += (int)r0[n] * k3[n];
                                        k3 += 16;
                                        sum3[n] += (int)r1[n] * k3[n];
                                        k3 += 16;
                                        sum3[n] += (int)r2[n] * k3[n];
                                        k3 += 16;
                                        sum3[n] += (int)r3[n] * k3[n];
                                        k3 -= 16 * 3;
                                    }
                                }

                                for (; q < inch; q++)
                                {
                                    const short *r0 = bottom_blob_tm_data + q * tiles * 16 + i * 16;
                                    const short *k0 = kernel0_tm + q * 16;
                                    const short *k1 = kernel1_tm + q * 16;
                                    const short *k2 = kernel2_tm + q * 16;
                                    const short *k3 = kernel3_tm + q * 16;

                                    for (int n = 0; n < 16; n++)
                                    {
                                        sum0[n] += (int)r0[n] * k0[n];
                                        sum1[n] += (int)r0[n] * k1[n];
                                        sum2[n] += (int)r0[n] * k2[n];
                                        sum3[n] += (int)r0[n] * k3[n];
                                    }
                                }

                                for (int n = 0; n < 16; n++)
                                {
                                    output0_tm[n] = sum0[n];
                                    output1_tm[n] = sum1[n];
                                    output2_tm[n] = sum2[n];
                                    output3_tm[n] = sum3[n];
                                }
#endif
                            }
                        }
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                        for (int p = remain_ou_start; p < ou; p++)
                        {
                            int *out0_tm = top_blob_tm_data + p * 16 * tiles;
                            const short *kernel0_tm = kernel_tm_data + p * 16 * inch;
                            for (int i = 0; i < tiles; i++)
                            {
                                int *output0_tm = out0_tm + i * 16;
                                int sum0[16] = {0};

                                int q = 0;
                                for (; q + 3 < inch; q += 4)
                                {
                                    const short *r0 = bottom_blob_tm_data + q * 16 * tiles + i * 16;
                                    const short *r1 = bottom_blob_tm_data + (q + 1) * 16 * tiles + i * 16;
                                    const short *r2 = bottom_blob_tm_data + (q + 2) * 16 * tiles + i * 16;
                                    const short *r3 = bottom_blob_tm_data + (q + 3) * 16 * tiles + i * 16;

                                    const short *k0 = kernel0_tm + q * 16;
                                    const short *k1 = kernel0_tm + (q + 1) * 16;
                                    const short *k2 = kernel0_tm + (q + 2) * 16;
                                    const short *k3 = kernel0_tm + (q + 3) * 16;

                                    for (int n = 0; n < 16; n++)
                                    {
                                        sum0[n] += r0[n] * k0[n];
                                        sum0[n] += r1[n] * k1[n];
                                        sum0[n] += r2[n] * k2[n];
                                        sum0[n] += r3[n] * k3[n];
                                    }
                                }
                                for (; q < inch; q++)
                                {
                                    const short *r0 = bottom_blob_tm_data + q * 16 * tiles + i * 16;
                                    const short *k0 = kernel0_tm + q * 16;
                                    for (int n = 0; n < 16; n++)
                                        sum0[n] += r0[n] * k0[n];
                                }
                                for (int n = 0; n < 16; n++)
                                    output0_tm[n] = sum0[n];
                            }
                        }
                    }
                    // END dot
                    // BEGIN transform output
                    int w_tm = outw >> 1 << 2;
                    int h_tm = outh >> 1 << 2;

                    int block_col = h_tm >> 2; // may be the block num in Feathercnn
                    int block_row = w_tm >> 2;

                    int *top_blob_tm_data = top_blob_tm->mutable_cpu_data();
                    const int tiles = block_col * block_row;
                    {

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                        for (int p = 0; p < ou; p++)
                        {
                            int *out_tm = top_blob_tm_data + p * 16 * tiles;
                            int *out = top_blob_bordered_data + n * ou * outh * outw + p * outh * outw;
                            //const float bias0 = bias ? bias[p] : 0.f;

                            for (int j = 0; j < block_col; j++)
                            {
                                int *outRow0 = out + 2 * j * outw;
                                int *outRow1 = out + (2 * j + 1) * outw;

                                for (int i = 0; i < block_row; i++)
                                {
                                    int *out_tile = out_tm + (j * block_row + i) * 16;
                                    int s0[4], s1[4], s2[4], s3[4], d0[2], d1[2], d2[2], d3[2];
                                    int w0[4], w1[4], o0[2], o1[2];
                                    for (int n = 0; n < 4; n++)
                                    {
                                        s0[n] = out_tile[n];
                                        s1[n] = out_tile[n + 4];
                                        s2[n] = out_tile[n + 8];
                                        s3[n] = out_tile[n + 12];
                                    }
                                    // w = A_T * W
                                    for (int n = 0; n < 4; n++)
                                    {
                                        w0[n] = s0[n] + s1[n] + s2[n];
                                        w1[n] = s1[n] - s2[n] + s3[n];
                                    }
                                    // save to top blob tm
                                    //outRow0[0] = w0[0] + w0[1] + w0[2] + bias0;
                                    //outRow0[1] = w1[0] + w1[1] + w1[2] + bias0;
                                    //outRow1[0] = w0[1] - w0[2] + w0[3] + bias0;
                                    //outRow1[1] = w1[1] - w1[2] + w1[3] + bias0;

                                    outRow0[0] = (w0[0] + w0[1] + w0[2]) >> 2;
                                    outRow0[1] = (w1[0] + w1[1] + w1[2]) >> 2;
                                    outRow1[0] = (w0[1] - w0[2] + w0[3]) >> 2;
                                    outRow1[1] = (w1[1] - w1[2] + w1[3]) >> 2;

                                    outRow0 += 2;
                                    outRow1 += 2;
                                }
                            }
                        }
                    }
                }

                //top_blob = top_blob_bordered_;
                cut_border_cpu<int>(top_blob_int8_bordered, top_blob, 0, top_blob_int8_bordered->height() - top_blob->height(), 0, top_blob_int8_bordered->width() - top_blob->width());
            }
        }

        template <typename Dtype>
        void operation_convolution<Dtype>::forward_im2col_int8_tr_kernel()
        {
            const int input_channel_ = this->weight_data_size_ * this->group_ / this->output_channel_ / this->kernel_size_h_ / this->kernel_size_w_;
            int inch = input_channel_ / this->group_;
            int outch = this->output_channel_ / this->group_;
            int kernel_size = this->kernel_size_h_ * this->kernel_size_w_;

            const signed char *kernel_base = this->weights_i8_[0]->cpu_data();
            // kernel memory packed 8 x 8
            int kernel_tm_channel = outch / 8 + (outch % 8) / 4 + outch % 4;

            int nn_outch = 0;
            int remain_outch_start = 0;
            nn_outch = outch >> 3;
            remain_outch_start = nn_outch << 3;

            for (int g = 0; g < this->group_; g++)
            {
                kernel_tm_int8_sgemm_.push_back(std::shared_ptr<memory::tensor<signed char>>(new memory::tensor<signed char>(std::vector<int>{1, kernel_tm_channel, inch, 8 * kernel_size}, this->params_.device_, memory::NCHW, nullptr)));
                signed char *kernel_tm_data = kernel_tm_int8_sgemm_[g]->mutable_cpu_data();
                int kernel_tm_cstep = kernel_tm_int8_sgemm_[g]->width() * kernel_tm_int8_sgemm_[g]->height();
                const signed char *kernel = kernel_base + this->weight_data_size_ * g / this->group_;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                for (int pp = 0; pp < nn_outch; pp++)
                {
                    int p = pp * 8;

                    const signed char *k0 = kernel + (p + 0) * inch * kernel_size;
                    const signed char *k1 = kernel + (p + 1) * inch * kernel_size;
                    const signed char *k2 = kernel + (p + 2) * inch * kernel_size;
                    const signed char *k3 = kernel + (p + 3) * inch * kernel_size;
                    const signed char *k4 = kernel + (p + 4) * inch * kernel_size;
                    const signed char *k5 = kernel + (p + 5) * inch * kernel_size;
                    const signed char *k6 = kernel + (p + 6) * inch * kernel_size;
                    const signed char *k7 = kernel + (p + 7) * inch * kernel_size;

                    signed char *ktmp = kernel_tm_data + (p / 8) * kernel_tm_cstep;
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

                        k0 += 1;
                        k1 += 1;
                        k2 += 1;
                        k3 += 1;
                        k4 += 1;
                        k5 += 1;
                        k6 += 1;
                        k7 += 1;
                    }
                }

                nn_outch = (outch - remain_outch_start) >> 2;
                for (int pp = 0; pp < nn_outch; pp++)
                {
                    int p = remain_outch_start + pp * 4;

                    const signed char *k0 = kernel + (p + 0) * inch * kernel_size;
                    const signed char *k1 = kernel + (p + 1) * inch * kernel_size;
                    const signed char *k2 = kernel + (p + 2) * inch * kernel_size;
                    const signed char *k3 = kernel + (p + 3) * inch * kernel_size;

                    signed char *ktmp = kernel_tm_data + (p / 8 + (p % 8) / 4) * kernel_tm_cstep;
                    for (int q = 0; q < inch * kernel_size; q++)
                    {
                        ktmp[0] = k0[0];
                        ktmp[1] = k1[0];
                        ktmp[2] = k2[0];
                        ktmp[3] = k3[0];
                        ktmp += 4;

                        k0 += 1;
                        k1 += 1;
                        k2 += 1;
                        k3 += 1;
                    }
                }
                remain_outch_start += nn_outch << 2;
                for (int p = remain_outch_start; p < outch; p++)
                {
                    const signed char *k0 = kernel + (p + 0) * inch * kernel_size;
                    signed char *ktmp = kernel_tm_data + (p / 8 + (p % 8) / 4 + p % 4) * kernel_tm_cstep;
                    for (int q = 0; q < inch * kernel_size; q++)
                    {
                        ktmp[0] = k0[0];
                        ktmp++;
                        k0++;
                    }
                }
            }
        }

        template <typename Dtype>
        void operation_convolution<Dtype>::conv_im2col_sgemm_int8_dequant_sse(const std::shared_ptr<memory::tensor<signed char>> &bottom_blob,
                                                                              std::shared_ptr<memory::tensor<float>> &top_blob, std::vector<float> &scale_dequants)
        {
            int w = bottom_blob->width();
            int h = bottom_blob->height();
            int inch = bottom_blob->channels() / this->group_;
            int bottom_cstep = w * h;
            int outw = top_blob->width();
            int outh = top_blob->height();
            int outch = top_blob->channels() / this->group_;

            const float *bias_0 = nullptr;
            if (this->bias_term_)
                bias_0 = this->weights_f32_[1]->cpu_data();

            float *top_data_base = top_blob->mutable_cpu_data();
            int top_cstep = top_blob->width() * top_blob->height();

            const signed char *bottom_data_base = bottom_blob->cpu_data();
            int out_size = outw * outh;
            int kernel_size = this->kernel_size_w_ * this->kernel_size_h_;
            int stride = this->kernel_size_h_ * this->kernel_size_w_ * outw * outh;

            for (int g = 0; g < this->group_; g++)
            {
                signed char *kernel_tm_data = kernel_tm_int8_sgemm_[g]->mutable_cpu_data();

                int kernel_tm_cstep = kernel_tm_int8_sgemm_[g]->width() * kernel_tm_int8_sgemm_[g]->height();
                const signed char *bottom_data = bottom_data_base + g * bottom_cstep * inch;
                const float *bias = bias_0 + g * outch;
                const float *scale_dequant = scale_dequants.data() + g * outch;
                float *top_data = top_data_base + out_size * outch * g;
                {
                    // im2row
                    bottom_im2col_int8_.reset(new memory::tensor<signed char>(std::vector<int>{1, 1, kernel_size * inch, out_size}, this->params_.device_, memory::NCHW, bottom_blob->allocator()));
                    signed char *ret = bottom_im2col_int8_->mutable_cpu_data();
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                    for (int p = 0; p < inch; p++)
                    {
                        const signed char *input = bottom_data + (p)*bottom_cstep;
                        int retID = stride * p;
                        for (int u = 0; u < this->kernel_size_h_; u++)
                        {
                            for (int v = 0; v < this->kernel_size_w_; v++)
                            {
                                for (int i = 0; i < outh; i++)
                                {
                                    for (int j = 0; j < outw; j++)
                                    {
                                        int row = u + i * this->stride_h_;
                                        int col = v + j * this->stride_w_;
                                        int index = row * w + col;
                                        ret[retID] = input[index];
                                        retID++;
                                    }
                                }
                            }
                        }
                    }

                    const signed char *bottom_im2col_data = bottom_im2col_int8_->cpu_data();

                    // int M = outch;  // outch
                    int N = outw * outh;                                        // outsize or out stride
                    int L = this->kernel_size_w_ * this->kernel_size_h_ * inch; // ksize * inch

                    // bottom_im2col_int8_ memory packed 8 x 8
                    bottom_tm_int8_.reset(new memory::tensor<signed char>(std::vector<int>{1, out_size / 8 + out_size % 8, inch, 8 * kernel_size}, this->params_.device_, memory::NCHW, bottom_blob->allocator()));
                    signed char *bottom_tm_data = bottom_tm_int8_->mutable_cpu_data();
                    int bottom_tm_cstep = bottom_tm_int8_->width() * bottom_tm_int8_->height();
                    {
                        int nn_size = out_size >> 3;
                        int remain_size_start = nn_size << 3;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                        for (int ii = 0; ii < nn_size; ii++)
                        {
                            int i = ii * 8;

                            const signed char *img0 = bottom_im2col_data;
                            img0 += i;
                            signed char *tmpptr = bottom_tm_data + (i / 8) * bottom_tm_cstep;

                            for (int q = 0; q < inch * kernel_size; q++)
                            {
#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX //AVX
                                _mm_storeu_si64(tmpptr, _mm_loadu_si64(img0));
#else
                                tmpptr[0] = img0[0];
                                tmpptr[1] = img0[1];
                                tmpptr[2] = img0[2];
                                tmpptr[3] = img0[3];
                                tmpptr[4] = img0[4];
                                tmpptr[5] = img0[5];
                                tmpptr[6] = img0[6];
                                tmpptr[7] = img0[7];
#endif
                                tmpptr += 8;
                                img0 += out_size;
                            }
                        }

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                        for (int i = remain_size_start; i < out_size; i++)
                        {
                            const signed char *img0 = bottom_im2col_data;
                            img0 += i;
                            signed char *tmpptr = bottom_tm_data + (i / 8 + i % 8) * bottom_tm_cstep;
                            for (int q = 0; q < inch * kernel_size; q++)
                            {
                                tmpptr[0] = img0[0];
                                tmpptr += 1;
                                img0 += out_size;
                            }
                        }
                    }
                    {
                        int nn_outch = 0;
                        int remain_outch_start = 0;

                        nn_outch = outch >> 3;
                        remain_outch_start = nn_outch << 3;

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                        for (int pp = 0; pp < nn_outch; pp++)
                        {
                            int i = pp * 8;

                            const float bias0 = bias ? bias[i] : 0.f;
                            const float bias1 = bias ? bias[i + 1] : 0.f;
                            const float bias2 = bias ? bias[i + 2] : 0.f;
                            const float bias3 = bias ? bias[i + 3] : 0.f;
                            const float bias4 = bias ? bias[i + 4] : 0.f;
                            const float bias5 = bias ? bias[i + 5] : 0.f;
                            const float bias6 = bias ? bias[i + 6] : 0.f;
                            const float bias7 = bias ? bias[i + 7] : 0.f;

                            const float scale_dequant0 = scale_dequant[i + 0];
                            const float scale_dequant1 = scale_dequant[i + 1];
                            const float scale_dequant2 = scale_dequant[i + 2];
                            const float scale_dequant3 = scale_dequant[i + 3];
                            const float scale_dequant4 = scale_dequant[i + 4];
                            const float scale_dequant5 = scale_dequant[i + 5];
                            const float scale_dequant6 = scale_dequant[i + 6];
                            const float scale_dequant7 = scale_dequant[i + 7];

                            float *output0 = top_data + (i)*top_cstep;
                            float *output1 = top_data + (i + 1) * top_cstep;
                            float *output2 = top_data + (i + 2) * top_cstep;
                            float *output3 = top_data + (i + 3) * top_cstep;
                            float *output4 = top_data + (i + 4) * top_cstep;
                            float *output5 = top_data + (i + 5) * top_cstep;
                            float *output6 = top_data + (i + 6) * top_cstep;
                            float *output7 = top_data + (i + 7) * top_cstep;

                            int j = 0;
                            for (; j + 7 < N; j = j + 8)
                            {
                                signed char *vb = bottom_tm_data + (j / 8) * bottom_tm_cstep;
                                signed char *va = kernel_tm_data + (i / 8) * kernel_tm_cstep;

#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX //AVX
                                __m256i _sum0 = _mm256_setzero_si256();
                                __m256i _sum1 = _mm256_setzero_si256();
                                __m256i _sum2 = _mm256_setzero_si256();
                                __m256i _sum3 = _mm256_setzero_si256();
                                __m256i _sum4 = _mm256_setzero_si256();
                                __m256i _sum5 = _mm256_setzero_si256();
                                __m256i _sum6 = _mm256_setzero_si256();
                                __m256i _sum7 = _mm256_setzero_si256();

                                __m256 _summm0 = _mm256_set1_ps(bias0);
                                __m256 _summm1 = _mm256_set1_ps(bias1);
                                __m256 _summm2 = _mm256_set1_ps(bias2);
                                __m256 _summm3 = _mm256_set1_ps(bias3);
                                __m256 _summm4 = _mm256_set1_ps(bias4);
                                __m256 _summm5 = _mm256_set1_ps(bias5);
                                __m256 _summm6 = _mm256_set1_ps(bias6);
                                __m256 _summm7 = _mm256_set1_ps(bias7);

                                __m256 scale0 = _mm256_set1_ps(scale_dequant0);
                                __m256 scale1 = _mm256_set1_ps(scale_dequant1);
                                __m256 scale2 = _mm256_set1_ps(scale_dequant2);
                                __m256 scale3 = _mm256_set1_ps(scale_dequant3);
                                __m256 scale4 = _mm256_set1_ps(scale_dequant4);
                                __m256 scale5 = _mm256_set1_ps(scale_dequant5);
                                __m256 scale6 = _mm256_set1_ps(scale_dequant6);
                                __m256 scale7 = _mm256_set1_ps(scale_dequant7);

                                int k = 0;
                                {
#include "operation_cxx/op_im2col_int8_pack8_sse1.cxx"
                                }
                                for (; k < L; k++)
                                {
                                    __m128i _va0 = _mm_set1_epi8(va[0]);
                                    __m128i _va1 = _mm_set1_epi8(va[1]);
                                    __m128i _va2 = _mm_set1_epi8(va[2]);
                                    __m128i _va3 = _mm_set1_epi8(va[3]);

                                    __m128i _va4 = _mm_set1_epi8(va[4]);
                                    __m128i _va5 = _mm_set1_epi8(va[5]);
                                    __m128i _va6 = _mm_set1_epi8(va[6]);
                                    __m128i _va7 = _mm_set1_epi8(va[7]);

                                    __m128i _vb0 = _mm_loadl_epi64((__m128i *)vb);

                                    _sum0 = _mm256_add_epi16_epi32(_vb0, _va0, _sum0);
                                    _sum1 = _mm256_add_epi16_epi32(_vb0, _va1, _sum1);
                                    _sum2 = _mm256_add_epi16_epi32(_vb0, _va2, _sum2);
                                    _sum3 = _mm256_add_epi16_epi32(_vb0, _va3, _sum3);
                                    _sum4 = _mm256_add_epi16_epi32(_vb0, _va4, _sum4);
                                    _sum5 = _mm256_add_epi16_epi32(_vb0, _va5, _sum5);
                                    _sum6 = _mm256_add_epi16_epi32(_vb0, _va6, _sum6);
                                    _sum7 = _mm256_add_epi16_epi32(_vb0, _va7, _sum7);

                                    va += 8;
                                    vb += 8;
                                }

                                __m256 _summ0 = _mm256_cvtepi32_ps(_sum0);
                                __m256 _summ1 = _mm256_cvtepi32_ps(_sum1);
                                __m256 _summ2 = _mm256_cvtepi32_ps(_sum2);
                                __m256 _summ3 = _mm256_cvtepi32_ps(_sum3);
                                __m256 _summ4 = _mm256_cvtepi32_ps(_sum4);
                                __m256 _summ5 = _mm256_cvtepi32_ps(_sum5);
                                __m256 _summ6 = _mm256_cvtepi32_ps(_sum6);
                                __m256 _summ7 = _mm256_cvtepi32_ps(_sum7);

                                _summm0 = _mm256_fmadd_ps(_summ0, scale0, _summm0);
                                _summm1 = _mm256_fmadd_ps(_summ1, scale1, _summm1);
                                _summm2 = _mm256_fmadd_ps(_summ2, scale2, _summm2);
                                _summm3 = _mm256_fmadd_ps(_summ3, scale3, _summm3);
                                _summm4 = _mm256_fmadd_ps(_summ4, scale4, _summm4);
                                _summm5 = _mm256_fmadd_ps(_summ5, scale5, _summm5);
                                _summm6 = _mm256_fmadd_ps(_summ6, scale6, _summm6);
                                _summm7 = _mm256_fmadd_ps(_summ7, scale7, _summm7);

                                _mm256_storeu_ps(output0, _summm0);
                                _mm256_storeu_ps(output1, _summm1);
                                _mm256_storeu_ps(output2, _summm2);
                                _mm256_storeu_ps(output3, _summm3);
                                _mm256_storeu_ps(output4, _summm4);
                                _mm256_storeu_ps(output5, _summm5);
                                _mm256_storeu_ps(output6, _summm6);
                                _mm256_storeu_ps(output7, _summm7);

#else
                                int sum0[8] = {0};
                                int sum1[8] = {0};
                                int sum2[8] = {0};
                                int sum3[8] = {0};
                                int sum4[8] = {0};
                                int sum5[8] = {0};
                                int sum6[8] = {0};
                                int sum7[8] = {0};

#include "operation_cxx/op_im2col_native1.cxx"

                                for (int n = 0; n < 8; n++)
                                {
                                    output0[n] = (float)sum0[n] * scale_dequant0 + bias0;
                                    output1[n] = (float)sum1[n] * scale_dequant1 + bias1;
                                    output2[n] = (float)sum2[n] * scale_dequant2 + bias2;
                                    output3[n] = (float)sum3[n] * scale_dequant3 + bias3;
                                    output4[n] = (float)sum4[n] * scale_dequant4 + bias4;
                                    output5[n] = (float)sum5[n] * scale_dequant5 + bias5;
                                    output6[n] = (float)sum6[n] * scale_dequant6 + bias6;
                                    output7[n] = (float)sum7[n] * scale_dequant7 + bias7;
                                }

#endif // __AVX__
                                output0 += 8;
                                output1 += 8;
                                output2 += 8;
                                output3 += 8;
                                output4 += 8;
                                output5 += 8;
                                output6 += 8;
                                output7 += 8;
                            }

                            for (; j < N; j++)
                            {
                                signed char *vb = bottom_tm_data + (j / 8 + j % 8) * bottom_tm_cstep;
                                ;
                                signed char *va = kernel_tm_data + (i / 8) * kernel_tm_cstep;

                                int sum0 = 0;
                                int sum1 = 0;
                                int sum2 = 0;
                                int sum3 = 0;
                                int sum4 = 0;
                                int sum5 = 0;
                                int sum6 = 0;
                                int sum7 = 0;

                                for (int k = 0; k < L; k++)
                                {
                                    sum0 += (int)va[0] * vb[0];
                                    sum1 += (int)va[1] * vb[0];
                                    sum2 += (int)va[2] * vb[0];
                                    sum3 += (int)va[3] * vb[0];
                                    sum4 += (int)va[4] * vb[0];
                                    sum5 += (int)va[5] * vb[0];
                                    sum6 += (int)va[6] * vb[0];
                                    sum7 += (int)va[7] * vb[0];

                                    va += 8;
                                    vb += 1;
                                }

                                output0[0] = (float)sum0 * scale_dequant0 + bias0;
                                output1[0] = (float)sum1 * scale_dequant1 + bias1;
                                output2[0] = (float)sum2 * scale_dequant2 + bias2;
                                output3[0] = (float)sum3 * scale_dequant3 + bias3;
                                output4[0] = (float)sum4 * scale_dequant4 + bias4;
                                output5[0] = (float)sum5 * scale_dequant5 + bias5;
                                output6[0] = (float)sum6 * scale_dequant6 + bias6;
                                output7[0] = (float)sum7 * scale_dequant7 + bias7;

                                output0++;
                                output1++;
                                output2++;
                                output3++;
                                output4++;
                                output5++;
                                output6++;
                                output7++;
                            }
                        }
                        nn_outch = (outch - remain_outch_start) >> 2;
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                        for (int pp = 0; pp < nn_outch; pp++)
                        {
                            int i = remain_outch_start + pp * 4;

                            float *output0 = top_data + (i)*top_cstep;
                            float *output1 = top_data + (i + 1) * top_cstep;
                            float *output2 = top_data + (i + 2) * top_cstep;
                            float *output3 = top_data + (i + 3) * top_cstep;

                            const float bias0 = bias ? bias[i] : 0.f;
                            const float bias1 = bias ? bias[i + 1] : 0.f;
                            const float bias2 = bias ? bias[i + 2] : 0.f;
                            const float bias3 = bias ? bias[i + 3] : 0.f;

                            const float scale_dequant0 = scale_dequant[i + 0];
                            const float scale_dequant1 = scale_dequant[i + 1];
                            const float scale_dequant2 = scale_dequant[i + 2];
                            const float scale_dequant3 = scale_dequant[i + 3];

                            int j = 0;
                            for (; j + 7 < N; j = j + 8)
                            {
                                signed char *vb = bottom_tm_data + (j / 8) * bottom_tm_cstep;
                                signed char *va = kernel_tm_data + (i / 8 + (i % 8) / 4) * kernel_tm_cstep;
#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX //AVX
                                __m256i _sum0 = _mm256_setzero_si256();
                                __m256i _sum1 = _mm256_setzero_si256();
                                __m256i _sum2 = _mm256_setzero_si256();
                                __m256i _sum3 = _mm256_setzero_si256();

                                __m256 _summm0 = _mm256_set1_ps(bias0);
                                __m256 _summm1 = _mm256_set1_ps(bias1);
                                __m256 _summm2 = _mm256_set1_ps(bias2);
                                __m256 _summm3 = _mm256_set1_ps(bias3);

                                __m256 scale0 = _mm256_set1_ps(scale_dequant0);
                                __m256 scale1 = _mm256_set1_ps(scale_dequant1);
                                __m256 scale2 = _mm256_set1_ps(scale_dequant2);
                                __m256 scale3 = _mm256_set1_ps(scale_dequant3);

                                int k = 0;
                                for (; k + 3 < L; k = k + 4)
                                {
                                    // k0
                                    __m128i _va0 = _mm_set1_epi8(va[0]);
                                    __m128i _va1 = _mm_set1_epi8(va[1]);
                                    __m128i _va2 = _mm_set1_epi8(va[2]);
                                    __m128i _va3 = _mm_set1_epi8(va[3]);

                                    __m128i _vb0 = _mm_loadl_epi64((__m128i *)vb);
                                    __m128i _vb1 = _mm_loadl_epi64((__m128i *)(vb + 8));
                                    __m128i _vb2 = _mm_loadl_epi64((__m128i *)(vb + 16));
                                    __m128i _vb3 = _mm_loadl_epi64((__m128i *)(vb + 24));

                                    _sum0 = _mm256_add_epi16_epi32(_vb0, _va0, _sum0);
                                    _sum1 = _mm256_add_epi16_epi32(_vb0, _va1, _sum1);
                                    _sum2 = _mm256_add_epi16_epi32(_vb0, _va2, _sum2);
                                    _sum3 = _mm256_add_epi16_epi32(_vb0, _va3, _sum3);

                                    va += 4;

                                    // k1
                                    _va0 = _mm_set1_epi8(va[0]);
                                    _va1 = _mm_set1_epi8(va[1]);
                                    _va2 = _mm_set1_epi8(va[2]);
                                    _va3 = _mm_set1_epi8(va[3]);

                                    _sum0 = _mm256_add_epi16_epi32(_vb1, _va0, _sum0);
                                    _sum1 = _mm256_add_epi16_epi32(_vb1, _va1, _sum1);
                                    _sum2 = _mm256_add_epi16_epi32(_vb1, _va2, _sum2);
                                    _sum3 = _mm256_add_epi16_epi32(_vb1, _va3, _sum3);

                                    va += 4;

                                    // k2
                                    _va0 = _mm_set1_epi8(va[0]);
                                    _va1 = _mm_set1_epi8(va[1]);
                                    _va2 = _mm_set1_epi8(va[2]);
                                    _va3 = _mm_set1_epi8(va[3]);

                                    _sum0 = _mm256_add_epi16_epi32(_vb2, _va0, _sum0);
                                    _sum1 = _mm256_add_epi16_epi32(_vb2, _va1, _sum1);
                                    _sum2 = _mm256_add_epi16_epi32(_vb2, _va2, _sum2);
                                    _sum3 = _mm256_add_epi16_epi32(_vb2, _va3, _sum3);

                                    va += 4;

                                    // k3
                                    _va0 = _mm_set1_epi8(va[0]);
                                    _va1 = _mm_set1_epi8(va[1]);
                                    _va2 = _mm_set1_epi8(va[2]);
                                    _va3 = _mm_set1_epi8(va[3]);

                                    _sum0 = _mm256_add_epi16_epi32(_vb3, _va0, _sum0);
                                    _sum1 = _mm256_add_epi16_epi32(_vb3, _va1, _sum1);
                                    _sum2 = _mm256_add_epi16_epi32(_vb3, _va2, _sum2);
                                    _sum3 = _mm256_add_epi16_epi32(_vb3, _va3, _sum3);

                                    va += 4;
                                    vb += 32;
                                }
                                for (; k < L; k++)
                                {
                                    // k0
                                    __m128i _va0 = _mm_set1_epi8(va[0]);
                                    __m128i _va1 = _mm_set1_epi8(va[1]);
                                    __m128i _va2 = _mm_set1_epi8(va[2]);
                                    __m128i _va3 = _mm_set1_epi8(va[3]);

                                    __m128i _vb0 = _mm_loadl_epi64((__m128i *)vb);

                                    _sum0 = _mm256_add_epi16_epi32(_vb0, _va0, _sum0);
                                    _sum1 = _mm256_add_epi16_epi32(_vb0, _va1, _sum1);
                                    _sum2 = _mm256_add_epi16_epi32(_vb0, _va2, _sum2);
                                    _sum3 = _mm256_add_epi16_epi32(_vb0, _va3, _sum3);

                                    va += 4;
                                    vb += 8;
                                }
                                __m256 _summ0 = _mm256_cvtepi32_ps(_sum0);
                                __m256 _summ1 = _mm256_cvtepi32_ps(_sum1);
                                __m256 _summ2 = _mm256_cvtepi32_ps(_sum2);
                                __m256 _summ3 = _mm256_cvtepi32_ps(_sum3);

                                _summm0 = _mm256_fmadd_ps(_summ0, scale0, _summm0);
                                _summm1 = _mm256_fmadd_ps(_summ1, scale1, _summm1);
                                _summm2 = _mm256_fmadd_ps(_summ2, scale2, _summm2);
                                _summm3 = _mm256_fmadd_ps(_summ3, scale3, _summm3);

                                _mm256_storeu_ps(output0, _summm0);
                                _mm256_storeu_ps(output1, _summm1);
                                _mm256_storeu_ps(output2, _summm2);
                                _mm256_storeu_ps(output3, _summm3);
#else
                                int sum0[8] = {0};
                                int sum1[8] = {0};
                                int sum2[8] = {0};
                                int sum3[8] = {0};
                                int k = 0;
                                for (; k + 7 < L; k = k + 8)
                                {
                                    for (int n = 0; n < 8; n++)
                                    {
                                        sum0[n] += va[0] * vb[n];
                                        sum1[n] += va[1] * vb[n];
                                        sum2[n] += va[2] * vb[n];
                                        sum3[n] += va[3] * vb[n];
                                        va += 4;

                                        sum0[n] += va[0] * vb[n + 8];
                                        sum1[n] += va[1] * vb[n + 8];
                                        sum2[n] += va[2] * vb[n + 8];
                                        sum3[n] += va[3] * vb[n + 8];
                                        va += 4;

                                        sum0[n] += va[0] * vb[n + 16];
                                        sum1[n] += va[1] * vb[n + 16];
                                        sum2[n] += va[2] * vb[n + 16];
                                        sum3[n] += va[3] * vb[n + 16];
                                        va += 4;

                                        sum0[n] += va[0] * vb[n + 24];
                                        sum1[n] += va[1] * vb[n + 24];
                                        sum2[n] += va[2] * vb[n + 24];
                                        sum3[n] += va[3] * vb[n + 24];
                                        va += 4;

                                        sum0[n] += va[0] * vb[n + 32];
                                        sum1[n] += va[1] * vb[n + 32];
                                        sum2[n] += va[2] * vb[n + 32];
                                        sum3[n] += va[3] * vb[n + 32];
                                        va += 4;

                                        sum0[n] += va[0] * vb[n + 40];
                                        sum1[n] += va[1] * vb[n + 40];
                                        sum2[n] += va[2] * vb[n + 40];
                                        sum3[n] += va[3] * vb[n + 40];
                                        va += 4;

                                        sum0[n] += va[0] * vb[n + 48];
                                        sum1[n] += va[1] * vb[n + 48];
                                        sum2[n] += va[2] * vb[n + 48];
                                        sum3[n] += va[3] * vb[n + 48];
                                        va += 4;

                                        sum0[n] += va[0] * vb[n + 56];
                                        sum1[n] += va[1] * vb[n + 56];
                                        sum2[n] += va[2] * vb[n + 56];
                                        sum3[n] += va[3] * vb[n + 56];
                                        va -= 28;
                                    }

                                    va += 32;
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
                                    }
                                    va += 4;
                                    vb += 8;
                                }
                                for (int n = 0; n < 8; n++)
                                {
                                    output0[n] = (float)sum0[n] * scale_dequant0 + bias0;
                                    output1[n] = (float)sum1[n] * scale_dequant1 + bias1;
                                    output2[n] = (float)sum2[n] * scale_dequant2 + bias2;
                                    output3[n] = (float)sum3[n] * scale_dequant3 + bias3;
                                }
#endif // __AVX__
                                output0 += 8;
                                output1 += 8;
                                output2 += 8;
                                output3 += 8;
                            }

                            for (; j < N; j++)
                            {
                                signed char *vb = bottom_tm_data + (j / 8 + j % 8) * bottom_tm_cstep;
                                ;
                                signed char *va = kernel_tm_data + (i / 8 + (i % 8) / 4) * kernel_tm_cstep;
                                int sum0 = 0;
                                int sum1 = 0;
                                int sum2 = 0;
                                int sum3 = 0;
                                for (int k = 0; k < L; k++)
                                {
                                    sum0 += va[0] * vb[0];
                                    sum1 += va[1] * vb[0];
                                    sum2 += va[2] * vb[0];
                                    sum3 += va[3] * vb[0];

                                    va += 4;
                                    vb += 1;
                                }
                                output0[0] = (float)sum0 * scale_dequant0 + bias0;
                                output1[0] = (float)sum1 * scale_dequant1 + bias1;
                                output2[0] = (float)sum2 * scale_dequant2 + bias2;
                                output3[0] = (float)sum3 * scale_dequant3 + bias3;

                                output0++;
                                output1++;
                                output2++;
                                output3++;
                            }
                        }
                        remain_outch_start += nn_outch << 2;
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
                        for (int i = remain_outch_start; i < outch; i++)
                        {
                            float *output = top_data + (i)*top_cstep;
                            const float bias0 = bias ? bias[i] : 0.f;
                            const float scale_dequant0 = scale_dequant[i];
                            int j = 0;
                            for (; j + 7 < N; j = j + 8)
                            {
                                signed char *vb = bottom_tm_data + (j / 8) * bottom_tm_cstep;
                                signed char *va = kernel_tm_data + (i / 8 + (i % 8) / 4 + i % 4) * kernel_tm_cstep;

#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX //AVX
                                __m256i _sum0 = _mm256_setzero_si256();
                                __m256 _summm0 = _mm256_set1_ps(bias0);
                                __m256 scale = _mm256_set1_ps(scale_dequant0);

                                int k = 0;
                                for (; k + 3 < L; k = k + 4)
                                {
                                    __m128i _va0 = _mm_set1_epi8(va[0]);
                                    __m128i _va1 = _mm_set1_epi8(va[1]);
                                    __m128i _va2 = _mm_set1_epi8(va[2]);
                                    __m128i _va3 = _mm_set1_epi8(va[3]);

                                    __m128i _vb0 = _mm_loadl_epi64((__m128i *)vb);
                                    __m128i _vb1 = _mm_loadl_epi64((__m128i *)(vb + 8));
                                    __m128i _vb2 = _mm_loadl_epi64((__m128i *)(vb + 16));
                                    __m128i _vb3 = _mm_loadl_epi64((__m128i *)(vb + 24));

                                    _sum0 = _mm256_add_epi16_epi32(_vb0, _va0, _sum0);
                                    _sum0 = _mm256_add_epi16_epi32(_vb1, _va0, _sum0);
                                    _sum0 = _mm256_add_epi16_epi32(_vb2, _va0, _sum0);
                                    _sum0 = _mm256_add_epi16_epi32(_vb3, _va0, _sum0);

                                    va += 4;
                                    vb += 32;
                                }
                                for (; k < L; k++)
                                {
                                    // k0
                                    __m128i _va0 = _mm_set1_epi8(va[0]);
                                    __m128i _vb0 = _mm_loadl_epi64((__m128i *)vb);
                                    _sum0 = _mm256_add_epi16_epi32(_vb0, _va0, _sum0);
                                    va += 1;
                                    vb += 8;
                                }
                                __m256 _summ0 = _mm256_cvtepi32_ps(_sum0);
                                _summm0 = _mm256_fmadd_ps(_summ0, scale, _summm0);

                                _mm256_storeu_ps(output, _summm0);
#else
                                float sum[8] = {0};

                                int k = 0;
                                for (; k + 7 < L; k = k + 8)
                                {
                                    for (int n = 0; n < 8; n++)
                                    {
                                        sum[n] += va[0] * vb[n];
                                        sum[n] += va[1] * vb[n + 8];
                                        sum[n] += va[2] * vb[n + 16];
                                        sum[n] += va[3] * vb[n + 24];
                                        sum[n] += va[4] * vb[n + 32];
                                        sum[n] += va[5] * vb[n + 40];
                                        sum[n] += va[6] * vb[n + 48];
                                        sum[n] += va[7] * vb[n + 56];
                                    }

                                    va += 8;
                                    vb += 64;
                                }

                                for (; k < L; k++)
                                {
                                    for (int n = 0; n < 8; n++)
                                    {
                                        sum[n] += va[0] * vb[n];
                                    }

                                    va += 1;
                                    vb += 8;
                                }

                                for (int n = 0; n < 8; n++)
                                {
                                    output[n] = sum[n] * scale_dequant0 + bias0;
                                }
#endif // __AVX__
                                output += 8;
                            }
                            for (; j < N; j++)
                            {
                                signed char *vb = bottom_tm_data + (j / 8 + j % 8) * bottom_tm_cstep;
                                ;
                                signed char *va = kernel_tm_data + (i / 8 + (i % 8) / 4 + i % 4) * kernel_tm_cstep;

                                int k = 0;
                                float sum0 = bias0;
                                for (; k < L; k++)
                                {
                                    sum0 += va[0] * vb[0];
                                    va += 1;
                                    vb += 1;
                                }
                                output[0] = sum0;
                                output++;
                            }
                        }
                    }
                }
            }
        }

        template <typename Dtype>
        void operation_convolution<Dtype>::forward_cpu_sbias(float *output, const float *bias, memory::orderType order)
        {
            if (order == memory::NCHW)
            {
                math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, this->output_channel_,
                                          this->output_spatial_dim_, 1, 1.0f, bias, bias_multiplier_data,
                                          1.0f, output);
            }
            else if (order == memory::NHWC)
            {
                math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, this->output_spatial_dim_,
                                          this->output_channel_, 1, 1.0f, bias_multiplier_data, bias,
                                          1.0f, output);
            }
            else
            {
                NOT_IMPLEMENTED;
            }
        }

        template <typename Dtype>
        void operation_convolution<Dtype>::forward_cpu_k1s1_f32(const std::shared_ptr<memory::tensor<float>> &bottom,
                                                                std::shared_ptr<memory::tensor<float>> &top)
        {
            float *top_data = top->mutable_cpu_data();
            const float *bottom_data = bottom->cpu_data();
            const float *weights_data = this->weights_f32_[0]->cpu_data();
            const int step = bottom->count(2, 4);
            memset(top_data, 0, top->count() * sizeof(float));

#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
            for (int i = 0; i < this->output_channel_; i++)
            {
                float *top_data_slice = top_data + i * step;
                for (int j = 0; j < this->input_channel_; j++)
                {
                    cblas_saxpy(step, weights_data[i * this->input_channel_ + j], bottom_data + j * step, 1, top_data_slice, 1);
                }
            }
            if (this->bias_term_)
            {
                forward_cpu_sbias(top_data, this->weights_f32_[1]->cpu_data(), memory::NCHW);
            }
        }

#ifndef USE_CUDA
        STUB_GPU(operation_convolution);
#endif

        INSTANCE_CLASS(operation_convolution);
        REGISTE(operation_convolution);
    }
}