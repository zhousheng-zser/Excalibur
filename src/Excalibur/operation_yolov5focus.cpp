#include "Excalibur/operation_yolov5focus.hpp"
#include "Excalibur/operation_reflector.hpp"
#include "Excalibur/math_functions.hpp"
#include <algorithm>
#include <cfloat>

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        operation_yolov5focus<Dtype>::operation_yolov5focus(const operation_param &param) : operation<Dtype>(param)
        {
        }

        template <class Dtype>
        void operation_yolov5focus<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            int num = bottoms[0]->num();
            int w = bottoms[0]->width();
            int h = bottoms[0]->height();
            int channels = bottoms[0]->channels();

            int outw = w / 2;
            int outh = h / 2;
            int outc = channels * 4;

            tops[0].reset(new memory::tensor<float>(std::vector<int>{num, outc, outh, outw}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));

            for (int p = 0; p < outc; p++)
            {
                const float *ptr = bottoms[0]->cpu_data() + bottoms[0]->offset(0, p % channels, (p / channels) % 2, (p / channels) / 2);
                float *outptr = tops[0]->mutable_cpu_data() + tops[0]->offset(0, p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        *outptr = *ptr;

                        outptr += 1;
                        ptr += 2;
                    }
                    ptr += w;
                }
            }
        }

        // #ifndef USE_CUDA
        //         STUB_GPU(operation_yolov5focus);
        // #endif

        template <typename Dtype>
        void operation_yolov5focus<Dtype>::forward_gpu_f32(
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

        INSTANCE_CLASS(operation_yolov5focus);
        REGISTE(operation_yolov5focus);
    }
}