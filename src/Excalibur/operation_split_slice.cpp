#include "Excalibur/operation_split_slice.hpp"
#include "Excalibur/operation_reflector.hpp"
#include "Excalibur/math_functions.hpp"
#include <algorithm>
#include <cfloat>

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        operation_split_slice<Dtype>::operation_split_slice(const operation_param& param) : operation<Dtype>(param)
        {
            std::vector<std::string> attrs = split_string(param.specific_params_, " ");
            for (int i = 0; i < attrs.size(); ++i)
            {
                std::vector<std::string> kvs = split_string(attrs[i], "=");
                switch (std::stoi(kvs[0]))
                {
                case 0:
                {
                    std::vector<std::string> sub_kvs = split_string(kvs[1], ",");
                    output_size_ = std::stoi(sub_kvs[0]);
                    for (size_t j = 0; j < output_size_; j++)
                        split_.push_back(std::stoi(sub_kvs[j + 1]));
                    break;
                }
                case 1:
                    axis_ = std::stoi(kvs[1]);
                    break;
                default:
                    LOG(FATAL) << "Un-supported Slice Attribution " << kvs[0];
                    break;
                }
            }
        }

        template <class Dtype>
        void operation_split_slice<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms, std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), output_size_);

            std::vector<int> shape = bottoms[0]->data_shape();
            int axis = axis_ + 1;
            CHECK_GT(axis, 0);
            CHECK_LT(axis, shape.size());
            int input_dim = shape[axis];
            split_[output_size_ - 1] = input_dim;

            for (size_t i = 0; i < output_size_ - 1; i++)
                split_[output_size_ - 1] -= split_[i];

            for (size_t i = 0, input_offset = 0; i < output_size_; i++)
            {
                int inner_size = 1;
                for (size_t j = axis + 1; j < shape.size(); ++j)
                {
                    inner_size *= shape[j];
                }
                int input_stride = input_dim * inner_size;

                std::vector<int> output_shape(shape);
                output_shape[axis] = split_[i];
                tops[i].reset(new memory::tensor<float>(output_shape, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
                int outer_size = 1;
                for (int j = 0; j < axis; ++j)
                {
                    outer_size *= shape[j];
                }
                int output_stride = split_[i] * inner_size;

                const float* input_data = bottoms[0]->cpu_data();
                float* output_data = tops[i]->mutable_cpu_data();

                if (bottoms[0]->order() == memory::NCHW)
                {
                    for (int j = 0; j < outer_size; ++j)
                    {
                        const float* input_base = input_data + j * input_stride + input_offset * inner_size;
                        float* output_base = output_data + j * output_stride;
                        math_functions::excalibur_copy(output_stride, input_base, output_base, tops[i]->device());
                    }

                    input_offset += split_[i];
                }
                else
                {
                    NOT_IMPLEMENTED;
                }
            }
        }

        template<typename Dtype>
        void operation_split_slice<Dtype>::forward_gpu_f32(
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

        INSTANCE_CLASS(operation_split_slice);
        REGISTE(operation_split_slice);
    }
}