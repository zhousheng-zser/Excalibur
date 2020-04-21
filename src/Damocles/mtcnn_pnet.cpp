#include "mtcnn_pnet.hpp"
#include "Primitives/memory.hpp"

#include <fstream>
#include <iostream>

using glasssix::memory::aligned_heap_free;

using namespace glasssix::memory;

namespace glasssix
{
	namespace longinus
	{
		mtcnn_pnet::mtcnn_pnet(int device)
		{
			device_ = device;
			if (device_ >= 0)
			{
				int8_quantization_ = false;//do not use int8 in GPU mode
			}

#ifdef __ARM_NEON
			int8_quantization_ = false;//do not use int8 in ARM mode
#endif

#if SIMD_TYPE >= SIMDTYPE_SSE
			//use for Copy_Int8_Params
			std::shared_ptr<memory::tensor<float>> bottom_round_ = std::make_shared<memory::tensor<float>>(std::vector<int>{mm_align_size});
			float* bottom_round_data_ = bottom_round_->mutable_cpu_data();
#endif // SIMD_TYPE >= SIMDTYPE_SSE

			float quantize_level = INT_MAX;
			if (int8_quantization_)
			{
				Copy_Int8_Params(conv1, mtcnn_pnet);
				Copy_Params(prelu1_weights, mtcnn_pnet, quantize_level);
				Copy_Int8_Params(conv2, mtcnn_pnet);
				Copy_Params(prelu2_weights, mtcnn_pnet, quantize_level);
				Copy_Int8_Params(conv3, mtcnn_pnet);
				Copy_Params(prelu3_weights, mtcnn_pnet, quantize_level);
				Copy_Int8_Params(conv4_1, mtcnn_pnet);
				Copy_Int8_Params(conv4_2, mtcnn_pnet);
			}
			else
			{
				//copy float32_data directly
				Copy_Params(conv1_weights, mtcnn_pnet, quantize_level);
				Copy_Params(conv1_bias, mtcnn_pnet, quantize_level);
				Copy_Params(prelu1_weights, mtcnn_pnet, quantize_level);
				Copy_Params(conv2_weights, mtcnn_pnet, quantize_level);
				Copy_Params(conv2_bias, mtcnn_pnet, quantize_level);
				Copy_Params(prelu2_weights, mtcnn_pnet, quantize_level);
				Copy_Params(conv3_weights, mtcnn_pnet, quantize_level);
				Copy_Params(conv3_bias, mtcnn_pnet, quantize_level);
				Copy_Params(prelu3_weights, mtcnn_pnet, quantize_level);
				Copy_Params(conv4_1_weights, mtcnn_pnet, quantize_level);
				Copy_Params(conv4_1_bias, mtcnn_pnet, quantize_level);
				Copy_Params(conv4_2_weights, mtcnn_pnet, quantize_level);
				Copy_Params(conv4_2_bias, mtcnn_pnet, quantize_level);
			}
			
#ifdef USE_CUDA
			CUBLAS_CHECK(cublasCreate(&cublas_handle_));
#ifdef USE_CUDNN
			CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
			cudnn_ready_ = true;
#endif
#endif

			//
#ifdef __ARM_NEON
			Init_Conv_arm_Params(conv1, 3, 10, 1, 3, 1, 0, true);
			Init_PReLU_arm_Params(prelu1, 10, false, false);
			Init_Pooling_arm_Params(pool1, 2, 2, 0, 0);
			Init_Conv_arm_Params(conv2, 10, 16, 1, 3, 1, 0, true);
			Init_PReLU_arm_Params(prelu2, 16, false, false);
			Init_Conv_arm_Params(conv3, 16, 32, 1, 3, 1, 0, true);
			Init_PReLU_arm_Params(prelu3, 32, false, false);
			Init_Conv_arm_Params(conv4_1, 32, 2, 1, 1, 1, 0, true);
			Init_Conv_arm_Params(conv4_2, 32, 4, 1, 1, 1, 0, true);
			Init_Softmax_arm_Params(prob1, 2);
#else
			Init_Conv_Params(conv1, 3, 10, 1, 3, 1, 0, true);
			Init_PReLU_Params(prelu1, 10, false);
			Init_Pooling_Params(pool1, 2, 2, 0, 0);
			Init_Conv_Params(conv2, 10, 16, 1, 3, 1, 0, true);
			Init_PReLU_Params(prelu2, 16, false);
			Init_Conv_Params(conv3, 16, 32, 1, 3, 1, 0, true);
			Init_PReLU_Params(prelu3, 32, false);
			Init_Conv_Params(conv4_1, 32, 2, 1, 1, 1, 0, true);
			Init_Conv_Params(conv4_2, 32, 4, 1, 1, 1, 0, true);
			Init_Softmax_Params(prob1, 2);
#endif
			//
		}


		mtcnn_pnet::~mtcnn_pnet()
		{
			delete conv1;
			delete prelu1;
			delete pool1;
			delete conv2;
			delete prelu2;
			delete conv3;
			delete prelu3;
			delete conv4_1;
			delete conv4_2;
			delete prob1;

			//conv_weights and bias free automatically, prelu_weights need to free explicitly
			aligned_heap_free(prelu1_weights);
			aligned_heap_free(prelu2_weights);
			aligned_heap_free(prelu3_weights);

#ifdef USE_CUDA
			if (cublas_handle_)
			{
				CUBLAS_CHECK(cublasDestroy(cublas_handle_));
			}
#ifdef USE_CUDNN
			if (cudnn_handle_)
			{
			    CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
			}
#endif
#endif
		}

		void mtcnn_pnet::Forward_cpu(const std::shared_ptr<memory::tensor<float>> input_data)
		{
			conv1->Forward(input_data, conv1_top_data);
			prelu1->Forward_cpu(conv1_top_data);
			pool1->Forward_cpu(conv1_top_data, pool1_top_data);
			conv2->Forward(pool1_top_data, conv2_top_data);
			prelu2->Forward_cpu(conv2_top_data);
			conv3->Forward(conv2_top_data, conv3_top_data);
			prelu3->Forward_cpu(conv3_top_data);
			conv4_1->Forward(conv3_top_data, conv4_1_top_data);
			conv4_2->Forward(conv3_top_data, conv4_2_top_data);
			prob1->Forward_cpu(conv4_1_top_data, prob1_top_data);
		}

#ifdef USE_CUDA
		void mtcnn_pnet::Forward_gpu_native(const std::shared_ptr<memory::tensor<float>> input_data)
		{
			conv1->Forward(cublas_handle_, input_data, conv1_top_data);
			prelu1->Forward_gpu_native(conv1_top_data);
			pool1->Forward_gpu_native(conv1_top_data, pool1_top_data);
			conv2->Forward(cublas_handle_, pool1_top_data, conv2_top_data);
			prelu2->Forward_gpu_native(conv2_top_data);
			conv3->Forward(cublas_handle_, conv2_top_data, conv3_top_data);
			prelu3->Forward_gpu_native(conv3_top_data);
			conv4_1->Forward(cublas_handle_, conv3_top_data, conv4_1_top_data);
			conv4_2->Forward(cublas_handle_, conv3_top_data, conv4_2_top_data);
			prob1->Forward_gpu_native(conv4_1_top_data, prob1_top_data);
		}

#ifdef USE_CUDNN
		void mtcnn_pnet::Forward_gpu_cudnn(const std::shared_ptr<memory::tensor<float>> input_data)
		{
			conv1->Forward(cudnn_handle_, input_data, conv1_top_data);
			prelu1->Forward_gpu_native(conv1_top_data);
			pool1->Forward_gpu_cudnn(conv1_top_data, pool1_top_data);
			conv2->Forward(cudnn_handle_, pool1_top_data, conv2_top_data);
			prelu2->Forward_gpu_native(conv2_top_data);
			conv3->Forward(cudnn_handle_, conv2_top_data, conv3_top_data);
			prelu3->Forward_gpu_native(conv3_top_data);
			conv4_1->Forward(cudnn_handle_, conv3_top_data, conv4_1_top_data);
			conv4_2->Forward(cudnn_handle_, conv3_top_data, conv4_2_top_data);
			prob1->Forward_gpu_cudnn(conv4_1_top_data, prob1_top_data);
		}

#endif
#endif

		void mtcnn_pnet::Forward(const std::shared_ptr<memory::tensor<float>> input_data)
		{
			if (device_<0)
			{
				Forward_cpu(input_data);
			}
			else
			{
#ifdef USE_CUDA
#ifdef USE_CUDNN
				Forward_gpu_cudnn(input_data);
				return;
#endif
				Forward_gpu_native(input_data);
				return;
#else
				NO_GPU;
#endif
			}
		}
	}
}
