#include "../../include/Excalibur/operation_deconvolution.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"
#include "../../include/Excalibur/math_functions.hpp"
#include "../../include/Excalibur/im2col.hpp"
#include "../../include/Excalibur/operation_make_border.hpp"
#include "../../include/Excalibur/operation_cut_border.hpp"
#include <random>

namespace glasssix
{
    namespace excalibur
    {
        template <typename Dtype>
        void operation_deconvolution<Dtype>::cut_padding(std::shared_ptr<memory::tensor<float>> &top_blob_bordered, std::shared_ptr<memory::tensor<float>> &top_blob)
        {
            if (this->pad_left_ > 0 || this->pad_right_ > 0 || this->pad_top_ > 0 || this->pad_bottom_ > 0)
            {
                std::shared_ptr<memory::tensor<float>> top_blob_bordered_adj = top_blob_bordered;
                if (this->output_pad_right_ > 0 || this->output_pad_bottom_ > 0)
                {
                    make_border(top_blob_bordered, top_blob_bordered_adj, 0, this->output_pad_bottom_, 0, this->output_pad_right_, border_constant, 0.f);
                    if (top_blob_bordered_adj->empty())
                    {
                        return;
                    }
                }

                cut_border_cpu(top_blob_bordered_adj, top_blob, this->pad_top_, this->pad_bottom_, this->pad_left_, this->pad_right_);
            }
            else if (this->output_dim_w_ > 0 && this->output_dim_h_ > 0)
            {
                std::shared_ptr<memory::tensor<float>> top_blob_bordered_adj = top_blob_bordered;
                if (this->output_pad_right_ > 0 || this->output_pad_bottom_ > 0)
                {
                    make_border(top_blob_bordered, top_blob_bordered_adj, 0, this->output_pad_bottom_, 0, this->output_pad_right_, border_constant, 0.f);
                    if (top_blob_bordered_adj->empty())
                        return;
                }

                int wcut = top_blob_bordered_adj->width() - this->output_dim_w_;
                int hcut = top_blob_bordered_adj->height() - this->output_dim_h_;

                if (this->pad_left_ == -233 || this->pad_right_ == -233 || this->pad_top_ == -233 || this->pad_bottom_ == -233)
                {
                    // onnx padding=SAME_UPPER
                    cut_border_cpu(top_blob_bordered_adj, top_blob, hcut / 2, hcut - hcut / 2, wcut / 2, wcut - wcut / 2);
                }
                else if (this->pad_left_ == -234 || this->pad_right_ == -234 || this->pad_top_ == -234 || this->pad_bottom_ == -234)
                {
                    // onnx padding=SAME_LOWER
                    cut_border_cpu(top_blob_bordered_adj, top_blob, hcut - hcut / 2, hcut / 2, wcut - wcut / 2, wcut / 2);
                }
            }
            else
            {
                if (this->output_pad_right_ > 0 || this->output_pad_bottom_ > 0)
                {
                    make_border(top_blob_bordered, top_blob, 0, this->output_pad_bottom_, 0, this->output_pad_right_, border_constant, 0.f);
                }
                else
                {
                    top_blob = top_blob_bordered;
                }
            }
        }

        template <typename Dtype>
        operation_deconvolution<Dtype>::operation_deconvolution(const operation_param &param) : operation_general_conv<Dtype>(param)
        {
        }

        template <typename Dtype>
        int operation_deconvolution<Dtype>::init_weights(FILE *fp)
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
                return mem;
            }
            else
            {
                NOT_IMPLEMENTED;
                return 0;
            }
        }

        template <typename Dtype>
        int operation_deconvolution<Dtype>::init_weights()
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
            }
            else
            {
                NOT_IMPLEMENTED;
            }
            return mem;
        }

        template <typename Dtype>
        void operation_deconvolution<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms,
                                                             std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            // CHECK_EQ(bottoms.size(), 1);
            // CHECK_EQ(tops.size(), 1);

            // this->num_ = bottoms[0]->data_shape()[0];
            // this->bottom_dim_ = bottoms[0]->count(1, 4);
            const float *bottom_data = bottoms[0]->cpu_data();
            // const float *weights = this->weights_f32_[0]->cpu_data();
            // const float *bias = nullptr;
            // if (bottoms[0]->order() != memory::NCHW)
            // {
            //     bottoms[0]->convert_order();
            // }
            // this->input_channel_ = bottoms[0]->channels();
            // this->input_dim_h_ = bottoms[0]->height();
            // this->input_dim_w_ = bottoms[0]->width();
            // this->input_spatial_dim_ = this->input_dim_h_ * this->input_dim_w_;
            // this->output_dim_h_ = (this->input_dim_h_ - 1) * this->stride_h_ + this->kernel_size_h_ - (this->pad_top_ + this->pad_bottom_);
            // this->output_dim_w_ = (this->input_dim_w_ - 1) * this->stride_w_ + this->kernel_size_w_ - (this->pad_left_ + this->pad_right_);
            // this->output_spatial_dim_ = this->output_dim_w_ * this->output_dim_h_;
            // tops[0].reset(new memory::tensor<float>(std::vector<int>{this->num_, this->output_channel_, this->output_dim_h_, this->output_dim_w_},
            //                                         bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            // float *top_data = tops[0]->mutable_cpu_data();
            // this->top_dim_ = tops[0]->count(1, 4);
            // col_buffer_.reset(new memory::tensor<float>(std::vector<int>{1, this->output_channel_ * this->kernel_size_h_ * this->kernel_size_w_, this->input_dim_h_, this->input_dim_w_},
            //                                             this->params_.device_, memory::NCHW, nullptr));
            // this->col_offset_ = this->input_spatial_dim_ * this->kernel_size_h_ * this->kernel_size_w_;
            // this->output_offset_ = this->output_channel_ * this->output_spatial_dim_ / this->group_;
            // if (this->bias_term_)
            // {
            //     this->bias_multiplier_.reset(new memory::tensor<float>(this->output_spatial_dim_, this->params_.device_, bottoms[0]->order(), nullptr));
            //     math_functions::cpu_set(this->output_dim_w_ * this->output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
            //     bias = this->weights_f32_[1]->cpu_data();
            // }
            // for (int n = 0; n < this->num_; n++)
            // {
            //     forward_sgemm(bottom_data + n * this->bottom_dim_, weights, top_data + n * this->top_dim_, bottoms[0]->order());
            //     if (this->bias_term_)
            //     {
            //         forward_sbias(top_data + n * this->top_dim_, bias, bottoms[0]->order());
            //     }
            // }

            /******************************************************/
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            int num = bottoms[0]->num();
            int num_output = this->output_channel_;
            int w = bottoms[0]->width();
            int h = bottoms[0]->height();
            int channels = bottoms[0]->channels();

            const int kernel_extent_w = this->dilation_w_ * (this->kernel_size_w_ - 1) + 1;
            const int kernel_extent_h = this->dilation_h_ * (this->kernel_size_h_ - 1) + 1;

            int outw = (w - 1) * this->stride_w_ + kernel_extent_w;
            int outh = (h - 1) * this->stride_h_ + kernel_extent_h;

            // Mat top_blob_bordered;
            std::shared_ptr<memory::tensor<float>> top_blob_bordered;
            // top_blob_bordered.create(outw, outh, num_output, elemsize, opt.workspace_allocator);
            top_blob_bordered.reset(new memory::tensor<float>(std::vector<int>{num, num_output, outh, outw}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            const int maxk = this->kernel_size_w_ * this->kernel_size_h_;

            // kernel offsets
            std::vector<int> _space_ofs(maxk);
            int *space_ofs = &_space_ofs[0];
            {
                int p1 = 0;
                int p2 = 0;
                int gap = outw * this->dilation_h_ - this->kernel_size_w_ * this->dilation_w_;
                for (int i = 0; i < this->kernel_size_h_; i++)
                {
                    for (int j = 0; j < this->kernel_size_w_; j++)
                    {
                        space_ofs[p1] = p2;
                        p1++;
                        p2 += this->dilation_w_;
                    }
                    p2 += gap;
                }
            }
            const float *weight_data = this->weights_f32_[0]->cpu_data();
            // num_output
            for (int p = 0; p < num_output; p++)
            {
                // Mat out = top_blob_bordered.channel(p);
                float *out_data = top_blob_bordered->mutable_cpu_data() + top_blob_bordered->offset(0, p);

                const float bias = this->bias_term_ ? this->weights_f32_[1]->cpu_data()[p] : 0.f;

                for (int i = 0; i < outw * outh; ++i)
                {
                    out_data[i] = bias;
                }

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        // float *outptr = out.row(i * this->stride_h_) + j * this->stride_w_;
                        float *outptr = out_data + i * this->stride_h_ * outw + j * this->stride_w_;

                        const float *kptr = (const float *)weight_data + maxk * channels * p;

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            // const Mat m = bottom_blob.channel(q);
                            const float *m_data = bottoms[0]->cpu_data() + bottoms[0]->offset(0, q);

                            // float val = *(m.row(i) + j);
                            float val = m_data[i * w + j];

                            for (int k = 0; k < maxk; k++)
                            {
                                float w = kptr[k];
                                outptr[space_ofs[k]] += val * w;
                            }

                            kptr += maxk;
                        }
                    }
                }
            }
            cut_padding(top_blob_bordered, tops[0]);
        }

        template <typename Dtype>
        void operation_deconvolution<Dtype>::forward_sgemm(const float *input, const float *weights, float *output, memory::orderType order)
        {
            col_buffer_data = col_buffer_->mutable_cpu_data();
            if (order == memory::NCHW)
            {
                for (int g = 0; g < this->group_; g++)
                {
                    math_functions::cpu_sgemm(CblasTrans, CblasNoTrans, this->output_channel_ * this->kernel_size_h_ * this->kernel_size_w_ / this->group_,
                                              this->input_spatial_dim_, this->input_channel_ / this->group_, 1.0f,
                                              weights + this->kernel_size_h_ * this->kernel_size_w_ * g,
                                              input + g * this->input_channel_ * this->input_spatial_dim_ / this->group_,
                                              0.0f, col_buffer_data + this->col_offset_ * g);
                }
                col2im_cpu(col_buffer_data, this->output_channel_, this->output_dim_h_, this->output_dim_w_, this->kernel_size_h_,
                           this->kernel_size_w_, this->pad_left_, this->pad_top_, this->stride_h_, this->stride_w_, this->dilation_h_, this->dilation_w_, output);
            }
            else if (order == memory::NHWC)
            {
                NOT_IMPLEMENTED;
                //if (this->group_ == 1)
                //{
                //	math_functions::cpu_sgemm(CblasNoTrans, CblasTrans, this->input_spatial_dim_, this->output_channel_ * kernel_length_,
                //		this->input_channel_, 1.0f,
                //		input, weights, 0.0f, col_buff);
                //}
                //else if (this->group_ > 1)
                //{
                //	float* temp = (float*)malloc(this->input_spatial_dim_ * kernel_length_ * sizeof(float));
                //	math_functions::cpu_sgemm(CblasNoTrans, CblasTrans, this->input_spatial_dim_, this->output_channel_ * kernel_length_ / this->group_,
                //		this->input_channel_, 1.0f,
                //		input, weights, 0.0f, temp);

                //	for (int i = 0; i < this->input_spatial_dim_; i++)
                //	{
                //		for (int j = 0; j < this->output_channel_; j++)
                //		{
                //			memcpy(col_buff + i * this->output_channel_ * kernel_length_ + j * kernel_length_,
                //				temp + i * kernel_length_, kernel_length_ * sizeof(float));
                //		}
                //	}

                //	free(temp);
                //}
                //else
                //{
                //	LOG(FATAL) << "illegal group!";
                //}

                //conv_col2im_cpu(col_buff, output);
            }
            else
            {
                NOT_IMPLEMENTED;
            }
        }

        template <typename Dtype>
        void operation_deconvolution<Dtype>::forward_sbias(float *output, const float *bias, memory::orderType order)
        {
            if (order == memory::NCHW)
            {
                math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, this->output_channel_,
                                          this->output_spatial_dim_, 1, 1.0f, bias, bias_multiplier_->cpu_data(),
                                          1.0f, output);
            }
            else if (order == memory::NHWC)
            {
                math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, this->output_spatial_dim_,
                                          this->output_channel_, 1, 1.0f, bias_multiplier_->cpu_data(), bias,
                                          1.0f, output);
            }
            else
            {
                NOT_IMPLEMENTED;
            }
        }

#ifndef USE_CUDA
        STUB_GPU(operation_deconvolution);
#endif

        INSTANCE_CLASS(operation_deconvolution);
        REGISTE(operation_deconvolution);
    }
}