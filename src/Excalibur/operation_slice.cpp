#include "Excalibur/operation_slice.hpp"
#include "Excalibur/operation_reflector.hpp"
#include "Excalibur/math_functions.hpp"
#include <algorithm>
#include <cfloat>

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        operation_slice<Dtype>::operation_slice(const operation_param &param) : start_(INT_MAX), end_(INT_MAX), axis_(INT_MAX), operation<Dtype>(param)
        {
            std::vector<std::string> attrs = split_string(param.specific_params_, " ");
            for (int i = 0; i < attrs.size(); ++i)
            {
                std::vector<std::string> kvs = split_string(attrs[i], "=");
                switch (std::stoi(kvs[0]))
                {
                case 0:
                    start_ = std::stoi(kvs[1]);
                    break;
                case 1:
                    end_ = std::stoi(kvs[1]);
                    break;
                case 2:
                    axis_ = std::stoi(kvs[1]);
                    break;
                default:
                    LOG(FATAL) << "Un-supported Slice Attribution " << kvs[0];
                    break;
                }
            }
            if (start_ == INT_MAX || end_ == INT_MAX || axis_ == INT_MAX)
            {
                LOG(FATAL) << "The starts/ends/axes are required.";
            }
        }

        template <class Dtype>
        void operation_slice<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            std::vector<int> shape = bottoms[0]->data_shape();
            CHECK_GE(axis_, 0);
            CHECK_LT(axis_, shape.size());
            int input_dim = shape[axis_];
            CHECK_GE(start_, 0);
            CHECK_LE(end_, input_dim);

            int output_dim = end_ - start_;
            int inner_size = 1;
            for (int i = axis_ + 1; i < shape.size(); ++i)
            {
                inner_size *= shape[i];
            }
            int input_stride = input_dim * inner_size;
            std::vector<int> output_shape(shape);
            output_shape[axis_] = output_dim;
            tops[0].reset(new memory::tensor<float>(output_shape, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            int outer_size = 1;
            for (int i = 0; i < axis_; ++i)
            {
                outer_size *= shape[i];
            }
            int output_stride = output_dim * inner_size;
            int offset = start_ * inner_size;
            const float *input_data = bottoms[0]->cpu_data();
            float *output_data = tops[0]->mutable_cpu_data();

            if (bottoms[0]->order() == memory::NCHW)
            {
                for (int i = 0; i < outer_size; ++i)
                {
                    const float *input_base = input_data + i * input_stride + offset;
                    float *output_base = output_data + i * output_stride;
                    memcpy(output_base, input_base, output_stride * sizeof(float));
                }
            }
            else
            {
                NOT_IMPLEMENTED;
            }
        }

#ifndef USE_CUDA
        STUB_GPU(operation_slice);
#endif

//         template<typename Dtype>
// 		void operation_slice<Dtype>::forward_gpu_f32(
// #ifdef USE_CUDA
// 			cublasHandle_t &cublas_handle_,
// #ifdef USE_CUDNN
// 			cudnnHandle_t cudnn_handle,
// #endif //!USE_CUDNN
// #endif //!USE_CUDA
// 			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
// 			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
// 		{
// 			forward_cpu_f32(bottoms, tops);
// 		}

        INSTANCE_CLASS(operation_slice);
        REGISTE(operation_slice);
    }
}