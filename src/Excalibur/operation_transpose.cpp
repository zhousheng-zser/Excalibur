#include "Excalibur/operation_transpose.hpp"
#include "Excalibur/operation_reflector.hpp"
#include "Excalibur/math_functions.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        operation_transpose<Dtype>::operation_transpose(const operation_param &param) : perms_{0, 1, 2}, operation<Dtype>(param)
        {
            std::vector<std::string> attrs = split_string(param.specific_params_, " ");
            for (int i = 0; i < attrs.size(); ++i)
            {
                std::vector<std::string> kvs = split_string(attrs[i], "=");
                switch (std::stoi(kvs[0]))
                {
                case 0:
                    perms_.clear();
                    for (std::string &v : split_string(kvs[1], ","))
                    {
                        this->perms_.push_back(std::stoi(v));
                    }
                    break;
                default:
                    LOG(FATAL) << "Un-supported Transpose Attribution " << kvs[0];
                    break;
                }
            }
        }

        template <class Dtype>
        void operation_transpose<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            int num = bottoms[0]->num();
            int width = bottoms[0]->width();
            int height = bottoms[0]->height();
            int channels = bottoms[0]->channels();

            if (bottoms[0]->order() == memory::NCHW)
            {
                if (perms_[0] == 0 && perms_[1] == 1 && perms_[2] == 2)
                {
                    // w h c
                    tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
                    tops[0] = bottoms[0];
                }
                else if (perms_[0] == 1 && perms_[1] == 0 && perms_[2] == 2)
                {
                    tops[0].reset(new memory::tensor<float>(std::vector<int>{num, channels, width, height}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
                    // h w c
                    for (int ch = 0; ch < channels; ++ch)
                    {
                        const float *ptr = bottoms[0]->cpu_data() + bottoms[0]->offset(0, ch);
                        float *outptr = tops[0]->mutable_cpu_data() + tops[0]->offset(0, ch);

                        for (int i = 0; i < width; ++i)
                        {
                            for (int j = 0; j < height; ++j)
                            {
                                outptr[i * height + j] = ptr[j * width + i];
                            }
                        }
                    }
                }
                else if (perms_[0] == 0 && perms_[1] == 2 && perms_[2] == 1)
                {
                    // w c h
                    tops[0].reset(new memory::tensor<float>(std::vector<int>{num, width, channels, height}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
                    for (int ch = 0; ch < height; ++ch)
                    {
                        float *outptr = tops[0]->mutable_cpu_data() + ch * channels * width;
                        for (int i = 0; i < channels; ++i)
                        {
                            const float *ptr = bottoms[0]->cpu_data() + i * width * height + ch * width;
                            for (int j = 0; j < width; ++j)
                            {
                                outptr[i * width + j] = ptr[j];
                            }
                        }
                    }
                }
                else if (perms_[0] == 2 && perms_[1] == 0 && perms_[2] == 1)
                {
                    // c w h
                    tops[0].reset(new memory::tensor<float>(std::vector<int>{num, width, channels, height}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));

                    for (int ch = 0; ch < height; ++ch)
                    {
                        float *outptr = tops[0]->mutable_cpu_data() + ch * width * channels;
                        for (int i = 0; i < width; ++i)
                        {
                            for (int j = 0; j < channels; ++j)
                            {
                                const float *ptr = bottoms[0]->cpu_data() + j * width * height + ch * width + i;
                                outptr[i * channels + j] = *ptr;
                            }
                        }
                    }
                }
                else if (perms_[0] == 1 && perms_[1] == 2 && perms_[2] == 0)
                {
                    // h c w
                    tops[0].reset(new memory::tensor<float>(std::vector<int>{num, width, channels, height}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));

                    for(int ch = 0; ch < width; ++ch)
                    {
                        float *outptr = tops[0]->mutable_cpu_data() + ch * height * channels;
                        for(int i = 0; i < channels; ++i)
                        {
                            const float *ptr = bottoms[0]->cpu_data() + i * width * height + ch;
                            for(int j = 0; j < height; ++j)
                            {
                                outptr[i * height + j] = ptr[j * width];
                            }
                        }
                    }
                }
                else if(perms_[0] == 2 && perms_[1] == 1 && perms_[2] == 0)
                {
                    // c h w
                    tops[0].reset(new memory::tensor<float>(std::vector<int>{num, channels, height, width}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));

                    for(int ch = 0; ch < width; ++ch)
                    {
                        float *outptr = tops[0]->mutable_cpu_data() + ch * height * channels;
                        for(int i = 0; i < height; ++i)
                        {
                            for(int j = 0; j < channels; ++j)
                            {
                                const float *ptr = bottoms[0]->cpu_data() + j * width * height + i * width + ch;
                                outptr[i * channels + j] = *ptr;
                            }
                        }
                    }
                }
            }
            else
            {
                NOT_IMPLEMENTED;
            }
        }

        template<typename Dtype>
		void operation_transpose<Dtype>::forward_gpu_f32(
#ifdef USE_CUDA
			cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			forward_cpu_f32(bottoms, tops);
		}

        INSTANCE_CLASS(operation_transpose);
        REGISTE(operation_transpose);
    }
}