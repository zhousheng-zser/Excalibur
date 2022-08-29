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
        operation_slice<Dtype>::operation_slice(const operation_param &param) : start_(INT_MAX), end_(INT_MAX), axis_(INT_MAX), step_(INT_MAX), operation<Dtype>(param)
        {
            std::vector<std::string> attrs = split_string(param.specific_params_, " ");
            for (int i = 0; i < attrs.size(); ++i)
            {
                std::vector<std::string> kvs = split_string(attrs[i], "=");
                std::vector<std::string> sub_kvs = split_string(kvs[1], ",");
                switch (std::stoi(kvs[0]))
                {
                case 0:
                    start_ = std::stoi(sub_kvs[1]);
                    break;
                case 1:
                    end_ = std::stoi(sub_kvs[1]);
                    break;
                case 2:
                    axis_ = std::stoi(sub_kvs[1]);
                    break;
                case 3:
                    step_ = std::stoi(sub_kvs[1]);
                    break;
                default:
                    LOG(FATAL) << "Un-supported Slice Attribution " << kvs[0];
                    break;
                }
            }
            if (start_ == INT_MAX || step_ == INT_MAX || axis_ == INT_MAX)
            {
                LOG(FATAL) << "The starts/step_/axes are required.";
            }
        }

        template <class Dtype>
        void operation_slice<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            std::vector<int> shape = bottoms[0]->data_shape();
            int axis = axis_ + 1;
            CHECK_GT(axis, 0);
            CHECK_LT(axis, shape.size());
            int input_dim = shape[axis];
            CHECK_GE(start_, 0);
            int end = end_ == INT_MAX ? input_dim : end_;
            CHECK_LE(end, input_dim);

            int output_dim = (end - start_) / step_;
            int inner_size = 1;
            for (int i = axis + 1; i < shape.size(); ++i)
            {
                inner_size *= shape[i];
            }
            int input_stride = input_dim * inner_size;
            std::vector<int> output_shape(shape);
            output_shape[axis] = output_dim;
            tops[0].reset(new memory::tensor<float>(output_shape, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            int outer_size = 1;
            for (int i = 0; i < axis; ++i)
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
                    if(step_ == 1)
                        memcpy(output_base, input_base, output_stride * sizeof(float));
                    else
                    {
                        for (size_t k = 0; k < output_stride; k++)
                        {
                            output_base[k] = input_base[k / inner_size * step_ * inner_size + k % inner_size];
                        }
                    }
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