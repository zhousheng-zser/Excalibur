#include "Excalibur/operation_shufflechannel.hpp"
#include "Excalibur/operation_reflector.hpp"
#include "Excalibur/math_functions.hpp"
#include <algorithm>
#include <cfloat>

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        operation_shufflechannel<Dtype>::operation_shufflechannel(const operation_param& param) : operation<Dtype>(param)
        {
            std::vector<std::string> attrs = split_string(param.specific_params_, " ");
            for (int i = 0; i < attrs.size(); ++i)
            {
                std::vector<std::string> kvs = split_string(attrs[i], "=");
                switch (std::stoi(kvs[0]))
                {
                case 0:
                    group_ = std::stoi(kvs[1]);
                    break;
                case 1:
                    reverse_ = std::stoi(kvs[1]);
                    break;
                default:
                    LOG(FATAL) << "Un-supported ShuffleChannel Attribution " << kvs[0];
                    break;
                }
            }
        }

        template <class Dtype>
        void operation_shufflechannel<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms, std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            int num = bottoms[0]->num();
            int w = bottoms[0]->width();
            int h = bottoms[0]->height();
            int channels = bottoms[0]->channels();

            CHECK_EQ(channels % group_, 0);

            int _group = reverse_ ? channels / group_ : group_;
            int channels_per_group = channels / _group;

            if (bottoms[0]->order() == memory::NHWC)
                NOT_IMPLEMENTED;

            tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));

            int c_step = w * h;
            for (size_t n = 0; n < num; n++)
            {
                const float* bottom_data = bottoms[0]->cpu_data();
                float* top_data = tops[0]->mutable_cpu_data();
                for (int i = 0; i < _group; i++)
                {
                    for (int j = 0; j < channels_per_group; j++)
                    {
                        int src_coffset = channels_per_group * i + j;
                        int dst_coffset = _group * j + i;
                        std::copy(bottom_data + src_coffset * c_step, bottom_data + (src_coffset + 1) * c_step, top_data + dst_coffset * c_step);
                    }
                }
            }
        }

#ifdef USE_CUDA
        template<typename Dtype>
        void operation_shufflechannel<Dtype>::forward_gpu_f32(
            cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
            cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
            const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
            std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            int num = bottoms[0]->num();
            int w = bottoms[0]->width();
            int h = bottoms[0]->height();
            int channels = bottoms[0]->channels();

            CHECK_EQ(channels % group_, 0);

            int _group = reverse_ ? channels / group_ : group_;
            int channels_per_group = channels / _group;

            if (bottoms[0]->order() == memory::NHWC)
                NOT_IMPLEMENTED;

            tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));

            int c_step = w * h;
            for (size_t n = 0; n < num; n++)
            {
                const float* bottom_data = bottoms[0]->gpu_data();
                float* top_data = tops[0]->mutable_gpu_data();
                for (int i = 0; i < _group; i++)
                {
                    for (int j = 0; j < channels_per_group; j++)
                    {
                        int src_coffset = channels_per_group * i + j;
                        int dst_coffset = _group * j + i;
                        cudaMemcpy(top_data + dst_coffset * c_step, bottom_data + src_coffset * c_step, sizeof(float) * c_step, cudaMemcpyDeviceToDevice);
                    }
                }
            }
        }
#endif

        INSTANCE_CLASS(operation_shufflechannel);
        REGISTE(operation_shufflechannel);
    }
}