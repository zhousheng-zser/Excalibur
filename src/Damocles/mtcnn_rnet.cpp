#include "mtcnn_rnet.hpp"
namespace glasssix
{
	namespace longinus
	{
		mtcnn_rnet::mtcnn_rnet(int device)
		{
			device_ = device;
			if (device_ >= 0)
			{
				int8_quantization_ = false;//do not use int8 in GPU mode
			}

#if SIMD_TYPE >= SIMDTYPE_SSE
			//use for Copy_Int8_Params
			std::shared_ptr<tensor<float>> bottom_round_ = std::make_shared<tensor<float>>(std::vector<int>{mm_align_size});
			float* bottom_round_data_ = bottom_round_->mutable_cpu_data();
#endif // SIMD_TYPE >= SIMDTYPE_SSE

			float quantize_level = INT_MAX;

			if (int8_quantization_)
			{
				Copy_Int8_Params(conv1, mtcnn_rnet);
				Copy_Params(prelu1_weights, mtcnn_rnet, quantize_level);
				Copy_Int8_Params(conv2, mtcnn_rnet);
				Copy_Params(prelu2_weights, mtcnn_rnet, quantize_level);
				Copy_Int8_Params(conv3, mtcnn_rnet);
				Copy_Params(prelu3_weights, mtcnn_rnet, quantize_level);
				Copy_Params(conv4_weights, mtcnn_rnet, quantize_level);
				Copy_Params(conv4_bias, mtcnn_rnet, quantize_level);
				Copy_Params(prelu4_weights, mtcnn_rnet, quantize_level);
				Copy_Params(conv5_1_weights, mtcnn_rnet, quantize_level);
				Copy_Params(conv5_1_bias, mtcnn_rnet, quantize_level);
				Copy_Params(conv5_2_weights, mtcnn_rnet, quantize_level);
				Copy_Params(conv5_2_bias, mtcnn_rnet, quantize_level);
				/*Copy_Params(conv5_3_weights, mtcnn_rnet, quantize_level);
				Copy_Params(conv5_3_bias, mtcnn_rnet, quantize_level);*/
			}
			else
			{
				//copy float32_data directly
				Copy_Params(conv1_weights, mtcnn_rnet, quantize_level);
				Copy_Params(conv1_bias, mtcnn_rnet, quantize_level);
				Copy_Params(prelu1_weights, mtcnn_rnet, quantize_level);
				Copy_Params(conv2_weights, mtcnn_rnet, quantize_level);
				Copy_Params(conv2_bias, mtcnn_rnet, quantize_level);
				Copy_Params(prelu2_weights, mtcnn_rnet, quantize_level);
				Copy_Params(conv3_weights, mtcnn_rnet, quantize_level);
				Copy_Params(conv3_bias, mtcnn_rnet, quantize_level);
				Copy_Params(prelu3_weights, mtcnn_rnet, quantize_level);
				Copy_Params(conv4_weights, mtcnn_rnet, quantize_level);
				Copy_Params(conv4_bias, mtcnn_rnet, quantize_level);
				Copy_Params(prelu4_weights, mtcnn_rnet, quantize_level);
				Copy_Params(conv5_1_weights, mtcnn_rnet, quantize_level);
				Copy_Params(conv5_1_bias, mtcnn_rnet, quantize_level);
				Copy_Params(conv5_2_weights, mtcnn_rnet, quantize_level);
				Copy_Params(conv5_2_bias, mtcnn_rnet, quantize_level);


				/*Copy_Params(conv5_3_weights, mtcnn_rnet, quantize_level);
				Copy_Params(conv5_3_bias, mtcnn_rnet, quantize_level);*/
			}
			
			//
#ifdef USE_CUDA
			CUBLAS_CHECK(cublasCreate(&cublas_handle_));
#ifdef USE_CUDNN
			CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
			cudnn_ready_ = true;
#endif
#endif
			//
#ifdef __ARM_NEON
			Init_Conv_arm_Params(conv1, 3, 28, 1, 3, 1, 0, true);
			Init_PReLU_arm_Params(prelu1, 28, false, false);
			Init_Pooling_arm_Params(pool1, 3, 2, 0, 0);
			Init_Conv_arm_Params(conv2, 28, 48, 1, 3, 1, 0, true);
			Init_PReLU_arm_Params(prelu2, 48, false, false);
			Init_Pooling_arm_Params(pool2, 3, 2, 0, 0);
			Init_Conv_arm_Params(conv3, 48, 64, 1, 2, 1, 0, true);
			Init_PReLU_arm_Params(prelu3, 64, false, false);
			Init_InnerProduct_arm_Params(conv4, 64, 3, 3, 128, true);
			Init_PReLU_arm_Params(prelu4, 128, false, false);
			Init_InnerProduct_arm_Params(conv5_1, 128, 1, 1, 2, true);
			Init_InnerProduct_arm_Params(conv5_2, 128, 1, 1, 4, true);
			Init_Softmax_arm_Params(prob1, 2);
#else
			Init_Conv_Params(conv1, 3, 28, 1, 3, 1, 0, true);
			Init_PReLU_Params(prelu1, 28, false);
			Init_Pooling_Params(pool1, 3, 2, 0, 0);
			Init_Conv_Params(conv2, 28, 48, 1, 3, 1, 0, true);
			Init_PReLU_Params(prelu2, 48, false);
			Init_Pooling_Params(pool2, 3, 2, 0, 0);
			Init_Conv_Params(conv3, 48, 64, 1, 2, 1, 0, true);
			Init_PReLU_Params(prelu3, 64, false);
			Init_InnerProduct_Params(conv4, 64, 3, 3, 128, true);
			Init_PReLU_Params(prelu4, 128, false);
			Init_InnerProduct_Params(conv5_1, 128, 1, 1, 2, true);
			Init_InnerProduct_Params(conv5_2, 128, 1, 1, 4, true);
			Init_Softmax_Params(prob1, 2);
#endif
		}


		mtcnn_rnet::~mtcnn_rnet()
		{
			delete conv1;
			delete prelu1;
			delete pool1;
			delete conv2;
			delete prelu2;
			delete pool2;
			delete conv3;
			delete prelu3;
			delete conv4;
			delete prelu4;
			delete conv5_1;
			delete conv5_2;
			delete prob1;

			//conv_weights and bias free automatically, prelu_weights need to free explicitly
			FreeHost(prelu1_weights, false);
			FreeHost(prelu2_weights, false);
			FreeHost(prelu3_weights, false);
			FreeHost(prelu4_weights, false);

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

		void mtcnn_rnet::Forward_cpu(const std::shared_ptr<tensor<float>> input_data)
		{
			conv1->Forward(input_data, conv1_top_data);
			prelu1->Forward_cpu(conv1_top_data);
			pool1->Forward_cpu(conv1_top_data, pool1_top_data);
			conv2->Forward(pool1_top_data, conv2_top_data);
			prelu2->Forward_cpu(conv2_top_data);
			pool2->Forward_cpu(conv2_top_data, pool2_top_data);
			conv3->Forward(pool2_top_data, conv3_top_data);
			prelu3->Forward_cpu(conv3_top_data);
			conv4->Forward_cpu(conv3_top_data, conv4_top_data);
			prelu4->Forward_cpu(conv4_top_data);
			conv5_1->Forward_cpu(conv4_top_data, conv5_1_top_data);
			conv5_2->Forward_cpu(conv4_top_data, conv5_2_top_data);
			prob1->Forward_cpu(conv5_1_top_data, prob1_top_data);
		}

#ifdef USE_CUDA
		void mtcnn_rnet::Forward_gpu_native(const std::shared_ptr<tensor<float>> input_data)
		{
			conv1->Forward(cublas_handle_, input_data, conv1_top_data);
			prelu1->Forward_gpu_native(conv1_top_data);
			pool1->Forward_gpu_native(conv1_top_data, pool1_top_data);
			conv2->Forward(cublas_handle_, pool1_top_data, conv2_top_data);
			prelu2->Forward_gpu_native(conv2_top_data);
			pool2->Forward_gpu_native(conv2_top_data, pool2_top_data);
			conv3->Forward(cublas_handle_, pool2_top_data, conv3_top_data);
			prelu3->Forward_gpu_native(conv3_top_data);
			conv4->Forward_gpu_native(cublas_handle_, conv3_top_data, conv4_top_data);
			prelu4->Forward_gpu_native(conv4_top_data);
			conv5_1->Forward_gpu_native(cublas_handle_, conv4_top_data, conv5_1_top_data);
			conv5_2->Forward_gpu_native(cublas_handle_, conv4_top_data, conv5_2_top_data);
			prob1->Forward_gpu_native(conv5_1_top_data, prob1_top_data);
		}
#ifdef USE_CUDNN
		void mtcnn_rnet::Forward_gpu_cudnn(const std::shared_ptr<tensor<float>> input_data)
		{
			conv1->Forward(cudnn_handle_, input_data, conv1_top_data);
			prelu1->Forward_gpu_native(conv1_top_data);
			pool1->Forward_gpu_cudnn(conv1_top_data, pool1_top_data);
			conv2->Forward(cudnn_handle_, pool1_top_data, conv2_top_data);
			prelu2->Forward_gpu_native(conv2_top_data);
			pool2->Forward_gpu_cudnn(conv2_top_data, pool2_top_data);
			conv3->Forward(cudnn_handle_, pool2_top_data, conv3_top_data);
			prelu3->Forward_gpu_native(conv3_top_data);
			conv4->Forward_gpu_native(cublas_handle_, conv3_top_data, conv4_top_data);
			prelu4->Forward_gpu_native(conv4_top_data);
			conv5_1->Forward_gpu_native(cublas_handle_, conv4_top_data, conv5_1_top_data);
			conv5_2->Forward_gpu_native(cublas_handle_, conv4_top_data, conv5_2_top_data);
			prob1->Forward_gpu_cudnn(conv5_1_top_data, prob1_top_data);
		}
#endif
#endif
		void mtcnn_rnet::Forward(const std::shared_ptr<tensor<float>> input_data)
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
