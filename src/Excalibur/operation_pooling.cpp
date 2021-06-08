#include "../../include/Excalibur/operation_pooling.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"
#include <cfloat>

namespace glasssix
{
    namespace excalibur
    {
        template <typename Dtype>
        operation_pooling<Dtype>::operation_pooling(const operation_param &param) : operation<Dtype>(param)
        {
            auto attrs = split_string(param.specific_params_, " ");
            for (size_t i = 0; i < attrs.size(); i++)
            {
                if (split_string(attrs[i], "=")[0] == "0")
                {
                    type_ = (pooling_type)atoi(split_string(attrs[i], "=")[1].c_str());
                }
                else if (split_string(attrs[i], "=")[0] == "1")
                {
                    kernel_size_w_ = atoi(split_string(attrs[i], "=")[1].c_str());
                    kernel_size_h_ = kernel_size_w_;
                }
                else if (split_string(attrs[i], "=")[0] == "2")
                {
                    stride_w_ = atoi(split_string(attrs[i], "=")[1].c_str());
                    stride_h_ = stride_w_;
                }
                else if (split_string(attrs[i], "=")[0] == "3")
                {
                    pad_left_ = atoi(split_string(attrs[i], "=")[1].c_str());
                    pad_right_ = pad_left_;
                    pad_top_ = pad_left_;
                    pad_bottom_ = pad_left_;
                }
                else if (split_string(attrs[i], "=")[0] == "4")
                {
                    global_pooling_ = (bool)atoi(split_string(attrs[i], "=")[1].c_str());
                }
                else if (split_string(attrs[i], "=")[0] == "5")
                {
                    pad_mode_ = atoi(split_string(attrs[i], "=")[1].c_str());
                }
                else if (split_string(attrs[i], "=")[0] == "11")
                {
                    kernel_size_h_ = atoi(split_string(attrs[i], "=")[1].c_str());
                }
                else if (split_string(attrs[i], "=")[0] == "12")
                {
                    stride_h_ = atoi(split_string(attrs[i], "=")[1].c_str());
                }
                else if (split_string(attrs[i], "=")[0] == "13")
                {
                    pad_top_ = atoi(split_string(attrs[i], "=")[1].c_str());
                }
                else if (split_string(attrs[i], "=")[0] == "14")
                {
                    pad_right_ = atoi(split_string(attrs[i], "=")[1].c_str());
                }
                else if (split_string(attrs[i], "=")[0] == "15")
                {
                    pad_bottom_ = atoi(split_string(attrs[i], "=")[1].c_str());
                }
                else if (split_string(attrs[i], "=")[0] == "-23330")
                {
                    //do nothing
                }
                else
                {
                    LOG(FATAL) << "Un-supported Pooling Attribution " << split_string(attrs[i], "=")[0];
                }
            }
        }

        template <typename Dtype>
        void operation_pooling<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms,
                                                       std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), tops.size());
            for (size_t i = 0; i < bottoms.size(); i++)
            {
                int num_ = bottoms[i]->num();
                int channels_ = bottoms[i]->channels();
                int height_ = bottoms[i]->height();
                int width_ = bottoms[i]->width();

                if (this->global_pooling_)
                {
                    this->kernel_size_w_ = width_;
                    this->kernel_size_h_ = height_;
                    this->pad_top_ = 0;
                    this->pad_bottom_ = 0;
                    this->pad_left_ = 0;
                    this->pad_right_ = 0;
                    this->stride_h_ = 1;
                    this->stride_w_ = 1;
                }
                int pooled_height_ = static_cast<int>(floor(static_cast<float>(
                                                               height_ + pad_top_ + pad_bottom_ - kernel_size_h_) /
                                                           stride_h_)) +
                                     1;
                int pooled_width_ = static_cast<int>(floor(static_cast<float>(
                                                              width_ + pad_left_ + pad_right_ - kernel_size_w_) /
                                                          stride_w_)) +
                                    1;
                if (bottoms[i]->order() == memory::NCHW)
                {
                    tops[i].reset(new memory::tensor<float>(std::vector<int>{num_, channels_, pooled_height_, pooled_width_},
                                                            bottoms[i]->device(), bottoms[i]->order(), bottoms[i]->allocator()));
                    const float *bottom_data = bottoms[i]->cpu_data();
                    float *top_data = tops[i]->mutable_cpu_data();
                    const int top_count = tops[i]->count(0, 4);

                    const int bottom_offset = bottoms[i]->offset(0, 1);
                    const int top_offset = tops[i]->offset(0, 1);
                    switch (type_)
                    {
                    case MAX:
                        for (int n = 0; n < num_; ++n)
                        {
                            for (int c = 0; c < channels_; ++c)
                            {
                                for (int ph = 0; ph < pooled_height_; ++ph)
                                {
                                    for (int pw = 0; pw < pooled_width_; ++pw)
                                    {
                                        int hstart = ph * stride_h_ - pad_top_;
                                        int wstart = pw * stride_w_ - pad_left_;
                                        int hend = std::min(hstart + kernel_size_h_, height_);
                                        int wend = std::min(wstart + kernel_size_w_, width_);
                                        hstart = std::max(hstart, 0);
                                        wstart = std::max(wstart, 0);
                                        float top_val = -FLT_MAX;
                                        for (int h = hstart; h < hend; ++h)
                                        {
                                            for (int w = wstart; w < wend; ++w)
                                            {
                                                const int index = h * width_ + w;
                                                top_val = std::max(top_val, bottom_data[index]);
                                            }
                                        }
                                        const int pool_index = ph * pooled_width_ + pw;
                                        top_data[pool_index] = top_val;
                                    }
                                }
                                // compute offset
                                bottom_data += bottom_offset;
                                top_data += top_offset;
                            }
                        }
                        break;
                    case AVE:
                        memset(top_data, 0, top_count * sizeof(float));
                        // The main loop
                        for (int n = 0; n < num_; ++n)
                        {
                            for (int c = 0; c < channels_; ++c)
                            {
                                for (int ph = 0; ph < pooled_height_; ++ph)
                                {
                                    for (int pw = 0; pw < pooled_width_; ++pw)
                                    {
                                        int hstart = ph * stride_h_ - pad_top_;
                                        int wstart = pw * stride_w_ - pad_left_;
                                        int hend = std::min(hstart + kernel_size_h_, height_ + pad_bottom_);
                                        int wend = std::min(wstart + kernel_size_w_, width_ + pad_right_);
                                        int pool_size = (hend - hstart) * (wend - wstart);
                                        hstart = std::max(hstart, 0);
                                        wstart = std::max(wstart, 0);
                                        hend = std::min(hend, height_);
                                        wend = std::min(wend, width_);
                                        for (int h = hstart; h < hend; ++h)
                                        {
                                            for (int w = wstart; w < wend; ++w)
                                            {
                                                top_data[ph * pooled_width_ + pw] += bottom_data[h * width_ + w];
                                            }
                                        }
                                        top_data[ph * pooled_width_ + pw] /= pool_size;
                                    }
                                }
                                // compute offset
                                bottom_data += bottom_offset;
                                top_data += top_offset;
                            }
                        }
                        break;
                    default:
                        LOG(FATAL) << "Unknown pooling method.";
                    }
                }
                else if (bottoms[i]->order() == memory::NHWC)
                {
                    tops[i].reset(new memory::tensor<float>(std::vector<int>{num_, pooled_height_, pooled_width_, channels_},
                                                            bottoms[i]->device(), bottoms[i]->order(), bottoms[i]->allocator()));
                    const float *bottom_data = bottoms[i]->cpu_data();
                    float *top_data = tops[i]->mutable_cpu_data();
                    const int top_count = tops[i]->count(0, 4);

                    switch (type_)
                    {
                    case MAX:
                        for (int n = 0; n < num_; ++n)
                        {
                            int top_index0 = n * pooled_width_ * pooled_height_ * channels_;
                            int bottom_index0 = n * width_ * height_ * channels_;
                            for (int ph = 0; ph < pooled_height_; ++ph)
                            {
                                int top_index1 = top_index0 + ph * pooled_width_ * channels_;
                                for (int pw = 0; pw < pooled_width_; ++pw)
                                {
                                    int top_index2 = top_index1 + pw * channels_;
                                    int hstart = ph * stride_h_ - pad_top_;
                                    int wstart = pw * stride_w_ - pad_left_;
                                    int hend = std::min(hstart + kernel_size_h_, height_);
                                    int wend = std::min(wstart + kernel_size_w_, width_);
                                    hstart = std::max(hstart, 0);
                                    wstart = std::max(wstart, 0);

                                    for (int c = 0; c < channels_; ++c)
                                    {
                                        float top_val = -FLT_MAX;
                                        for (int h = hstart; h < hend; ++h)
                                        {
                                            int bottom_index1 = bottom_index0 + h * width_ * channels_;
                                            for (int w = wstart; w < wend; ++w)
                                            {
                                                int bottom_index2 = bottom_index1 + w * channels_;
                                                top_val = std::max(top_val, bottom_data[bottom_index2 + c]);
                                            }
                                        }
                                        top_data[top_index2 + c] = top_val;
                                    }
                                }
                            }
                        }
                        break;
                    case AVE:
                        memset(top_data, 0, top_count * sizeof(float));
                        // The main loop
                        for (int n = 0; n < num_; ++n)
                        {
                            int top_index0 = n * pooled_width_ * pooled_height_ * channels_;
                            int bottom_index0 = n * width_ * height_ * channels_;
                            for (int ph = 0; ph < pooled_height_; ++ph)
                            {
                                int top_index1 = top_index0 + ph * pooled_width_ * channels_;
                                for (int pw = 0; pw < pooled_width_; ++pw)
                                {
                                    int top_index2 = top_index1 + pw * channels_;
                                    int hstart = ph * stride_h_ - pad_top_;
                                    int wstart = pw * stride_w_ - pad_left_;
                                    int hend = std::min(hstart + kernel_size_h_, height_ + pad_bottom_);
                                    int wend = std::min(wstart + kernel_size_w_, width_ + pad_right_);
                                    int pool_size = (hend - hstart) * (wend - wstart);
                                    hstart = std::max(hstart, 0);
                                    wstart = std::max(wstart, 0);
                                    hend = std::min(hend, height_);
                                    wend = std::min(wend, width_);

                                    for (int c = 0; c < channels_; ++c)
                                    {
                                        for (int h = hstart; h < hend; ++h)
                                        {
                                            int bottom_index1 = bottom_index0 + h * width_ * channels_;
                                            for (int w = wstart; w < wend; ++w)
                                            {
                                                int bottom_index2 = bottom_index1 + w * channels_;
                                                top_data[top_index2 + c] += bottom_data[bottom_index2 + c];
                                            }
                                        }
                                        top_data[top_index2 + c] /= pool_size;
                                    }
                                }
                            }
                        }
                        break;
                    default:
                        LOG(FATAL) << "Unknown pooling method.";
                    }
                }
                else
                {
                    LOG(FATAL) << "Un-supported Order Type.";
                }
            }
        }

#ifndef USE_CUDA
        STUB_GPU(operation_pooling);
#endif

        INSTANCE_CLASS(operation_pooling);
        REGISTE(operation_pooling);
    }
}