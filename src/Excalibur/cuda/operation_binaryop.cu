// #include "../../../include/Excalibur/operation_binaryop.hpp"
// #include "../../../include/Excalibur/operation_reflector.hpp"

// namespace glasssix
// {
//     namespace excalibur
//     {
// #ifdef USE_CUDA

//         __global__ void ADD(int N, const float *input0, const float *input1, float *output, int mod_val)
//         {
//             CUDA_KERNEL_LOOP(i, N)
//             {
//                 int k = i % mod_val;
//                 output[i] = input0[i] + input1[k];
//             }
//         }

//         __global__ void SUB(int N, const float *input0, const float *input1, float *output, int mod_val)
//         {
//             CUDA_KERNEL_LOOP(i, N)
//             {
//                 output[i] = input0[i] - input1[i];
//             }
//         }

//         __global__ void MUL(int N, const float *input0, const float *input1, float *output, int mod_val)
//         {
//             CUDA_KERNEL_LOOP(i, N)
//             {
//                 output[i] = input0[i] * input1[i];
//             }
//         }

//         __global__ void DIV(int N, const float *input0, const float *input1, float *output, int mod_val)
//         {
//             CUDA_KERNEL_LOOP(i, N)
//             {
//                 output[i] = input0[i] / input1[i];
//             }
//         }

//         __global__ void add_scalar(int N, const float *input0, float coef, float *output)
//         {
//             CUDA_KERNEL_LOOP(i, N)
//             {
//                 output[i] = input0[i] + coef;
//             }
//         }

//         __global__ void sub_scalar(int N, const float *input0, float coef, float *output)
//         {
//             CUDA_KERNEL_LOOP(i, N)
//             {
//                 output[i] = input0[i] - coef;
//             }
//         }

//         __global__ void mul_scalar(int N, const float *input0, float coef, float *output)
//         {
//             CUDA_KERNEL_LOOP(i, N)
//             {
//                 output[i] = input0[i] * coef;
//             }
//         }

//         __global__ void div_scalar(int N, const float *input0, float coef, float *output)
//         {
//             CUDA_KERNEL_LOOP(i, N)
//             {
//                 output[i] = input0[i] / coef;
//             }
//         }

//         template <class Dtype>
//         void operation_binaryop<Dtype>::forward_gpu_f32(
//             cublasHandle_t &cublas_handle_,
// #ifdef USE_CUDNN
//             cudnnHandle_t cudnn_handle,
// #endif
//             const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms,
//             std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
//         {
//             CHECK_GE(bottoms.size(), 1);
//             CHECK_EQ(tops.size(), 1);

//             int count0 = bottoms[0]->count();
//             int count1 = bottoms[0]->count();
//             const float *input0 = bottoms[0]->gpu_data();
//             const float *input1 = bottoms[1]->gpu_data();
//             tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
//             float *output = tops[0]->mutable_gpu_data();

//             if (bottoms[0]->order() == memory::NCHW)
//             {
// #define COMPUTE_FLOAT(TYPE)                                                                           \
//     if (op_type_ == Operation_##TYPE)                                                                 \
//     {                                                                                                 \
//         int mod_val = count0 == count1 ? 1 : bottoms[1]->width();                                     \
//         TYPE<<<CUDA_GET_BLOCKS(count0), CUDA_NUM_THREADS>>>(count0, input0, input1, output, mod_val); \
//     };

//                 COMPUTE_FLOAT(ADD);
//                 COMPUTE_FLOAT(SUB);
//                 COMPUTE_FLOAT(MUL);
//                 COMPUTE_FLOAT(DIV);
//             }
//             else
//             {
//                 NOT_IMPLEMENTED;
//             }
//         }

// #ifdef USE_CUDNN
//         INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_binaryop);
// #else
//         INSTANTIATE_OPERATION_CUDA_FWDF32(operation_binaryop);
// #endif

// #endif
//     }
// }