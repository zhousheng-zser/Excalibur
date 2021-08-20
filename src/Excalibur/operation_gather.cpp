#include "Excalibur/operation.hpp"
#include "Excalibur/operation_gather.hpp"
#include "Excalibur/operation_reflector.hpp"
#include "Primitives/pool_allocator.hpp"
#include "Excalibur/math_functions.hpp"

#include <algorithm>
#include <numeric>

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        operation_gather<Dtype>::operation_gather(const operation_param &param) : operation<Dtype>(param), axis_(0)
        {
            std::vector<std::string> attrs = split_string(param.specific_params_, " ");
            for (int i = 0; i < attrs.size(); ++i)
            {
                std::vector<std::string> kvs = split_string(attrs[i], "=");
                switch (std::stoi(kvs[0]))
                {
                case 0:
                    //axis
                    axis_ = std::stoi(kvs[1]);
                    break;
                case 1:
                    for (std::string &i : split_string(kvs[1], ","))
                    {
                        indexs_.push_back(std::stoi(i));
                    }
                    break;
                default:
                    LOG(FATAL) << "Un-supported gather Attribution " << kvs[1];
                    break;
                }
            }
        }

        template <class Dtype>
        void operation_gather<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            if (bottoms[0]->order() == memory::NCHW)
            {
                std::vector<int> input_shape = bottoms[0]->data_shape();
                int input_rank = input_shape.size();
                int M = std::accumulate(input_shape.begin(), input_shape.begin() + axis_, 1, std::multiplies<int>());
                int N = indexs_.size();
                int block = std::accumulate(input_shape.begin() + axis_ + 1, input_shape.end(), 1, std::multiplies<int>());
                int data_batch = std::accumulate(input_shape.begin() + axis_, input_shape.end(), 1, std::multiplies<int>());
                int generate_batch = N * block;
                std::vector<int> output_shape;
                output_shape.reserve(4);
                output_shape.insert(output_shape.end(), input_shape.begin(), input_shape.begin() + axis_);
                output_shape.insert(output_shape.end(), input_shape.begin() + axis_ + 1, input_shape.end());
                output_shape.insert(output_shape.end(), N);
                tops[0].reset(new memory::tensor<float>(output_shape, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));

                const float *src_base = bottoms[0]->cpu_data();
                float *dst_base = tops[0]->mutable_cpu_data();

                auto lambda = [&](int index)
                {
                    int batch = index / N;
                    int i = index % N;
                    int src_offset_batch = batch * data_batch;
                    int dst_offset_batch = batch * generate_batch;
                    int idx = indexs_[i];
                    int src_offset = src_offset_batch + idx * block;
                    int dst_offset = dst_offset_batch + i * block;
                    memcpy(dst_base + dst_offset, src_base + src_offset, block * sizeof(float));
                };
                for (int i = 0; i < M * N; ++i)
                {
                    lambda(i);
                }
            }
            else
            {
                NOT_IMPLEMENTED;
            }
        }

        template <typename Dtype>
        void operation_gather<Dtype>::forward_gpu_f32(
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

        INSTANCE_CLASS(operation_gather);
        REGISTE(operation_gather);
    }
}