#include "Excalibur/operation_interp.hpp"
#include "Excalibur/math_functions.hpp"
#include "Excalibur/operation_reflector.hpp"
#include "Excalibur/operation_resize.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        operation_interp<Dtype>::operation_interp(const operation_param &param) : operation<Dtype>(param)
        {
            std::vector<std::string> attrs = split_string(param.specific_params_, " ");
            for (int i = 0; i < attrs.size(); ++i)
            {
                std::vector<std::string> kvs = split_string(attrs[i], "=");
                switch (std::stoi(kvs[0]))
                {
                case 0:
                    this->resize_type_ = std::stoi(kvs[1]);
                    break;
                case 1:
                    this->width_scale_ = std::stof(kvs[1]);
                    break;
                case 2:
                    this->height_scale_ = std::stof(kvs[1]);
                    break;
                case 3:
                    this->output_width_ = std::stoi(kvs[1]);
                    break;
                case 4:
                    this->output_height_ = std::stoi(kvs[1]);
                    break;
                case 5:
                    this->dynamic_target_size_ = std::stoi(kvs[1]);
                    break;
                case 6:
                    this->align_corner_ = std::stoi(kvs[1]);
                    break;
                default:
                    LOG(FATAL) << "Un-supported Interp Attribution " << kvs[0];
                    break;
                }
            }
        }

        template <class Dtype>
        void operation_interp<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            int num = bottoms[0]->num();
            int bottom_w = bottoms[0]->width();
            int bottom_h = bottoms[0]->height();
            int bottom_c = bottoms[0]->channels();

            int outw = output_width_;
            int outh = output_height_;

            if (output_width_ == 0 || output_height_ == 0)
            {
                outw = static_cast<int>(bottom_w * width_scale_);
                outh = static_cast<int>(bottom_h * height_scale_);
            }

            if (bottom_h == 1 && bottom_c == 1)
            {
                tops[0].reset(new memory::tensor<float>(std::vector<int>{num, bottom_w, outh, outw}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
                const float *bottom_data = bottoms[0]->cpu_data();
                int size = outw * outh;
                for (int n = 0; n < num; ++n)
                {
                    for (int q = 0; q < bottom_w; ++q)
                    {
                        float *top_data = tops[0]->mutable_cpu_data() + tops[0]->offset(n, q);
                        for (int i = 0; i < size; ++i)
                        {
                            top_data[i] = bottom_data[q];
                        }
                    }
                }
                return;
            }

            if (bottom_w == outw && bottom_h == outh)
            {
                tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
                tops[0] = bottoms[0];
                return;
            }

            tops[0].reset(new memory::tensor<float>(std::vector<int>{num, bottom_c, outh, outw}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));

            if (resize_type_ == 1) // nearest
            {
                const float hs = outh ? bottom_h / (float)outh : 1.f / height_scale_;
                const float ws = outw ? bottom_w / (float)outw : 1.f / width_scale_;

                for (int q = 0; q < bottom_c; q++)
                {
                    const float *ptr = bottoms[0]->cpu_data() + bottoms[0]->offset(0, q);
                    float *outptr = tops[0]->mutable_cpu_data() + tops[0]->offset(0, q);
                    for (int y = 0; y < outh; y++)
                    {
                        int in_y = std::min((int)(y * hs), (bottom_h - 1));
                        for (int x = 0; x < outw; x++)
                        {
                            int in_x = std::min((int)(x * ws), (bottom_w - 1));
                            *outptr++ = ptr[in_y * bottom_w + in_x];
                        }
                    }
                }
                return;
            }

            resize_cpu(bottoms[0], tops[0], outh, outw, interpolationType(resize_type_ - 1));
        }

        template <class Dtype>
        void operation_interp<Dtype>::forward_gpu_f32(
#ifdef USE_CUDA
                cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
                cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
                const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms,
                std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
                {
                    forward_cpu_f32(bottoms, tops);
                }

        INSTANCE_CLASS(operation_interp);
        REGISTE(operation_interp);
    }
}