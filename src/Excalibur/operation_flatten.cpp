#include "Excalibur/operation.hpp"
#include "Excalibur/operation_flatten.hpp"
#include "Excalibur/operation_reflector.hpp"
#include "Primitives/pool_allocator.hpp"
#include "Excalibur/math_functions.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        operation_flatten<Dtype>::operation_flatten(const operation_param &param) : operation<Dtype>(param)
        {
            std::vector<std::string> attrs = split_string(param.specific_params_, " ");
            for (int i = 0; i < attrs.size(); ++i)
            {
                std::vector<std::string> kvs = split_string(attrs[i], "=");
                switch (std::stoi(kvs[0]))
                {
                case 0:
                    //axis
                    this->axis_ = std::stoi(kvs[1]);
                    break;
                default:
                    LOG(FATAL) << "Un-supported flatten Attribution " << kvs[1];
                    break;
                }
            }
        }

        template <class Dtype>
        void operation_flatten<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            int blob_w = bottoms[0]->width();
            int blob_h = bottoms[0]->height();
            int blob_c = bottoms[0]->channels();

            int w, h;
            if (this->axis_ == 0)
            {
                w = 1;
                h = blob_w * blob_h * blob_c;
            }
            else if (this->axis_ == 1)
            {
                w = blob_w;
                h = blob_h * blob_c;
            }
            else if (this->axis_ == 2)
            {
                w = blob_w * blob_h;
                h = blob_c;
            }
            else if (this->axis_ = 3)
            {
                w = blob_w * blob_h * blob_c;
                h = 1;
            }
            else
            {
                NOT_IMPLEMENTED;
            }

            tops[0].reset(new memory::tensor<float>(std::vector<int>{bottoms[0]->num(), 1, h, w}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            math_functions::excalibur_copy(bottoms[0]->count(), bottoms[0]->cpu_data(), tops[0]->mutable_cpu_data(), bottoms[0]->device());
        }

        template<typename Dtype>
        void operation_flatten<Dtype>::forward_gpu_f32(
#ifdef USE_CUDA
            cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
            cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
            const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
            std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            int blob_w = bottoms[0]->width();
            int blob_h = bottoms[0]->height();
            int blob_c = bottoms[0]->channels();

            int w, h;
            if (this->axis_ == 0)
            {
                w = 1;
                h = blob_w * blob_h * blob_c;
            }
            else if (this->axis_ == 1)
            {
                w = blob_w;
                h = blob_h * blob_c;
            }
            else if (this->axis_ == 2)
            {
                w = blob_w * blob_h;
                h = blob_c;
            }
            else if (this->axis_ = 3)
            {
                w = blob_w * blob_h * blob_c;
                h = 1;
            }
            else
            {
                NOT_IMPLEMENTED;
            }

            tops[0].reset(new memory::tensor<float>(std::vector<int>{bottoms[0]->num(), 1, h, w}, this->params_.device_, bottoms[0]->order(), bottoms[0]->allocator()));
            math_functions::excalibur_copy(bottoms[0]->count(), bottoms[0]->gpu_data(), tops[0]->mutable_gpu_data(), this->params_.device_);
        }

        INSTANCE_CLASS(operation_flatten);
        REGISTE(operation_flatten);
    }
}