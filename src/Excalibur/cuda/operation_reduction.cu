#include "../../../include/Excalibur/operation_reduction.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
    namespace excalibur
    {
#ifdef USE_CUDA
        template <typename T>
        struct reduction_op_sumsq
        {
            __host__ __device__ T operator()(const T& x, const T& y) const
            {
                return x + y * y;
            }
        };

        template <typename T>
        struct reduction_op_add
        {
            __host__ __device__ T operator()(const T& x, const T& y) const
            {
                return x + y;
            }
        };

        template <typename T>
        struct post_process_sqrt
        {
            __host__ __device__ T operator()(const T& x) const
            {
                return (T)(sqrt(x));
            }
        };

        __global__ void ReduceMeanForward(
            const int nthreads,
            const float* const bottom_data,
            float* const top_data,
            const int channels,
            const int steps)
        {
            CUDA_KERNEL_LOOP(index, nthreads)
            {
				float sum = 0;
				for (size_t c = 0; c < channels; ++c) {
					sum += bottom_data[c * steps + index];
				}
                top_data[index] = sum / channels;
            }
        }

        template <typename Op>
        __global__ void op_keepdims_kernel(int N, float v0, const float* bottom_data, float* top_data)
        {
            Op op;
            CUDA_KERNEL_LOOP(i, N)
            {
                top_data[i] = op(v0, bottom_data[i]);
            }
        }

        template<size_t WARP_SIZE>
        __global__ void reduction_warp_keepdims_kernel(int N, const float* src, float* dst)
        {
            const size_t global_id = threadIdx.x + blockDim.x * blockIdx.x;
            const size_t reduce_id = global_id % WARP_SIZE;
            float val = global_id < N ? src[global_id] : 0;
            for (size_t offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
                val += __shfl_xor_sync(0xffffffff, val, offset, WARP_SIZE);
            if (reduce_id == 0)
                dst[global_id / WARP_SIZE] = val;
        }

        template <size_t WARP_SIZE, typename Op, typename Op2>
        static void reduction_op_keepdims(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top, float v0, bool reduce_w, bool reduce_h, bool reduce_c)
        {
            int width = bottom->width();
            int height = bottom->height();
            int channels = bottom->channels();
            int dims = (width == 1 ? 0 : 1) + (height == 1 ? 0 : 1) + (channels == 1 ? 0 : 1);
            if (dims == 0) dims = 1;
            if (dims == 1)
            {
                top.reset(new memory::tensor<float>(bottom->data_shape(), bottom->device(), bottom->order(), bottom->allocator()));
                for (size_t num = 0; num < bottom->num(); num++)
                {
                    const float* bottom_data = bottom->gpu_data() + bottom->offset(num);
                    float* top_data = top->mutable_gpu_data() + top->offset(num);

                    op_keepdims_kernel<Op> << <CUDA_GET_BLOCKS(width), CUDA_NUM_THREADS >> > (width, v0, bottom_data, top_data);

                    float* src_data = top_data;
                    float* dst_data = top->mutable_gpu_data() + num;

                    for (int len = (width + WARP_SIZE - 1) & ~(WARP_SIZE - 1), N = width;
                        len >= WARP_SIZE && N > 1;
                        N = len / WARP_SIZE, len = (len / WARP_SIZE + WARP_SIZE - 1) & ~(WARP_SIZE - 1))
                        dst_data += len / WARP_SIZE;

                    for (int len = (width + WARP_SIZE - 1) & ~(WARP_SIZE - 1), N = width;
                        len >= WARP_SIZE && N > 1;
                        N = len / WARP_SIZE, len = (len / WARP_SIZE + WARP_SIZE - 1) & ~(WARP_SIZE - 1))
                    {
                        dst_data -= len / WARP_SIZE;
                        reduction_warp_keepdims_kernel<WARP_SIZE> << <CUDA_GET_BLOCKS(len), CUDA_NUM_THREADS >> > (N, src_data, dst_data);
                        src_data = dst_data;
                    }
                }
                if(top->order() == memory::NCHW)
                    top->reshape(std::vector<int>{bottom->num(), channels, height, 1});
                else
                    top->reshape(std::vector<int>{bottom->num(), height, 1, channels});
            }
        }

        template<typename MathOp>
        __global__ void post_process_kernel(int N, float* top_data)
        {
            MathOp op;
            CUDA_KERNEL_LOOP(i, N)
            {
                top_data[i] = op(top_data[i]);
            }
        }

        template <typename MathOp>
        static void reduction_post_process(std::shared_ptr<memory::tensor<float>>& top)
        {
            int width = top->width();
            int height = top->height();
            int channels = top->channels();
            int dims = (width == 1 ? 0 : 1) + (height == 1 ? 0 : 1) + (channels == 1 ? 0 : 1);
            if (dims == 0) dims = 1;
            float* top_data = top->mutable_gpu_data();
            if (dims == 1)
            {
                post_process_kernel<MathOp> << <CUDA_GET_BLOCKS(width*top->num()), CUDA_NUM_THREADS >> > (width * top->num(), top_data);
            }
        }

        template <typename Op, typename Op2, typename Op3>
        static void reduction(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top, float v0, bool reduce_w, bool reduce_h, bool reduce_c, bool post_process, int keepdims)
        {
            if (keepdims)
            {
                reduction_op_keepdims<32, Op, Op2>(bottom, top, v0, reduce_w, reduce_h, reduce_c);
            }
            else
            {
                // reduction_op<Op, Op2>(a, b, v0, reduce_w, reduce_h, reduce_c, opt);
            }

            if (post_process)
            {
                reduction_post_process<Op3>(top);
            }
        }

        static void reduction_mean(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top)
        {
            //auto shape = bottom->data_shape();
            //int channels_num = shape[1];
            //int steps = shape[2] * shape[3];

            //shape[1] = 1; //reset outTensor channels
            //top.reset(new memory::tensor<float>(shape, bottom->device(), bottom->order(), bottom->allocator()));
            //float* bottom_data = bottom->mutable_gpu_data();
            //float* top_data = top->mutable_gpu_data();
            //const int top_count = top->count();
            //ReduceMeanForward << <CUDA_GET_BLOCKS(top_count), CUDA_NUM_THREADS >> > (top_count, bottom_data, top_data, channels_num, steps);
            auto shape = bottom->data_shape();
            int NUM = shape[0];
            int Channels = shape[1];
            int Height = shape[2];
            int Width = shape[3];
            int HWsize = Height * Width;
            int bottomCHWstp = Channels * Height * Width;

            shape[1] = 1; //reset outTensor channels
            top.reset(new memory::tensor<float>(shape, bottom->device(), bottom->order(), bottom->allocator()));
            int topCHWstp = top->channels() * top->height() * top->width(); //top_count

            float* top_data = top->mutable_cpu_data();
            float* bottom_data = bottom->mutable_cpu_data();

            for (int num = 0; num < NUM; ++num) { // BatchNum
                float* top_data_num = top_data + num * topCHWstp;
                const float* bottom_data_num = bottom_data + num * bottomCHWstp;
                ReduceMeanForward << <CUDA_GET_BLOCKS(topCHWstp), CUDA_NUM_THREADS >> > (topCHWstp, bottom_data_num, top_data_num, Channels, HWsize);

            }
        }

        template <class Dtype>
        void operation_reduction<Dtype>::forward_gpu_f32(
            cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
            cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
            const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
            std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            bool reduce_w = false;
            bool reduce_h = false;
            bool reduce_c = false;

            int width = bottoms[0]->width();
            int height = bottoms[0]->height();
            int channels = bottoms[0]->channels();
            int dims = (width == 1 ? 0 : 1) + (height == 1 ? 0 : 1) + (channels == 1 ? 0 : 1);

            if (dims == 1)
            {
                reduce_w = true;
            }

            if (operation_ == ReductionOp_L2)
                reduction<reduction_op_sumsq<float>, reduction_op_add<float>, post_process_sqrt<float>>(bottoms[0], tops[0], 0.f, reduce_w, reduce_h, reduce_c, true, keepdims_);
            else if (operation_ == ReductionOp_MEAN)
                reduction_mean(bottoms[0], tops[0]); //support 3dims only
            else
                NOT_IMPLEMENTED;
        }

#ifdef USE_CUDNN
        INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_reduction);
#else
        INSTANTIATE_OPERATION_CUDA_FWDF32(operation_reduction);
#endif

#endif
    }
}