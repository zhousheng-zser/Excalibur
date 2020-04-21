#include "nose_nir_net.hpp"
#include "nose_nir_net_data.hpp"
#include "Primitives/memory.hpp"

using glasssix::memory::aligned_heap_free;

namespace glasssix
{
	namespace longinus
	{
		Nose_nir_net::Nose_nir_net(int device)
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
			//copy float32_data directly
			Copy_Params(conv1_weights, nose_nir_net, quantize_level);
			Copy_Params(conv1_bias, nose_nir_net, quantize_level);
			Copy_Params(prelu1_weights, nose_nir_net, quantize_level);
			Copy_Params(conv1_dw_weights, nose_nir_net, quantize_level);
			Copy_Params(conv1_dw_bias, nose_nir_net, quantize_level);
			Copy_Params(prelu1_dw_weights, nose_nir_net, quantize_level);
			Copy_Params(conv2_weights, nose_nir_net, quantize_level);
			Copy_Params(conv2_bias, nose_nir_net, quantize_level);
			Copy_Params(conv2_dw_weights, nose_nir_net, quantize_level);
			Copy_Params(conv2_dw_bias, nose_nir_net, quantize_level);
			Copy_Params(prelu2_dw_weights, nose_nir_net, quantize_level);
			Copy_Params(conv3_weights, nose_nir_net, quantize_level);
			Copy_Params(conv3_bias, nose_nir_net, quantize_level);
			Copy_Params(conv3_dw_weights, nose_nir_net, quantize_level);
			Copy_Params(conv3_dw_bias, nose_nir_net, quantize_level);
			Copy_Params(prelu3_dw_weights, nose_nir_net, quantize_level);
			Copy_Params(conv4_weights, nose_nir_net, quantize_level);
			Copy_Params(conv4_bias, nose_nir_net, quantize_level);
			Copy_Params(conv4_dw_weights, nose_nir_net, quantize_level);
			Copy_Params(conv4_dw_bias, nose_nir_net, quantize_level);
			Copy_Params(prelu4_dw_weights, nose_nir_net, quantize_level);
			Copy_Params(conv5_weights, nose_nir_net, quantize_level);
			Copy_Params(conv5_bias, nose_nir_net, quantize_level);
			Copy_Params(prelu5_weights, nose_nir_net, quantize_level);
			Copy_Params(conv6_1_weights, nose_nir_net, quantize_level);
			Copy_Params(conv6_1_bias, nose_nir_net, quantize_level);


			//			
#ifdef USE_CUDA
			CUBLAS_CHECK(cublasCreate(&cublas_handle_));
#ifdef USE_CUDNN
			CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
			cudnn_ready_ = true;
#endif
#endif

#ifdef __ARM_NEON
			Init_Conv_arm_Params(conv1, 1, 16, 1, 3, 2, 1, true);
			Init_PReLU_arm_Params(prelu1, 16, false, false);
			Init_Conv_arm_Params(conv1_dw, 16, 16, 16, 3, 1, 1, true);
			Init_PReLU_arm_Params(prelu1_dw, 16, false, false);
			Init_Conv_arm_Params(conv2, 16, 32, 1, 1, 1, 0, true);
			Init_Conv_arm_Params(conv2_dw, 32, 32, 32, 3, 2, 1, true);
			Init_PReLU_arm_Params(prelu2_dw, 32, false, false);
			Init_Conv_arm_Params(conv3, 32, 32, 1, 1, 1, 0, true);
			Init_Conv_arm_Params(conv3_dw, 32, 32, 32, 3, 2, 1, true);
			Init_PReLU_arm_Params(prelu3_dw, 32, false, false);
			Init_Conv_arm_Params(conv4, 32, 64, 1, 1, 1, 0, true);
			Init_Conv_arm_Params(conv4_dw, 64, 64, 64, 3, 2, 1, true);
			Init_PReLU_arm_Params(prelu4_dw, 64, false, false);
			Init_InnerProduct_arm_Params(conv5, 64, 3, 3, 256, true);
			Init_PReLU_arm_Params(prelu5, 256, false, false);
			Init_InnerProduct_arm_Params(conv6_1, 256, 1, 1, 1, true);
			Init_Sigmoid_arm_Params(cls_loss);
#else
			Init_Conv_Params(conv1, 1, 16, 1, 3, 2, 1, true);
			Init_PReLU_Params(prelu1, 16, false);
			Init_Conv_Params(conv1_dw, 16, 16, 16, 3, 1, 1, true);
			Init_PReLU_Params(prelu1_dw, 16, false);
			Init_Conv_Params(conv2, 16, 32, 1, 1, 1, 0, true);
			Init_Conv_Params(conv2_dw, 32, 32, 32, 3, 2, 1, true);
			Init_PReLU_Params(prelu2_dw, 32, false);
			Init_Conv_Params(conv3, 32, 32, 1, 1, 1, 0, true);
			Init_Conv_Params(conv3_dw, 32, 32, 32, 3, 2, 1, true);
			Init_PReLU_Params(prelu3_dw, 32, false);
			Init_Conv_Params(conv4, 32, 64, 1, 1, 1, 0, true);
			Init_Conv_Params(conv4_dw, 64, 64, 64, 3, 2, 1, true);
			Init_PReLU_Params(prelu4_dw, 64, false);
			Init_InnerProduct_Params(conv5, 64, 3, 3, 256, true);
			Init_PReLU_Params(prelu5, 256, false);
			Init_InnerProduct_Params(conv6_1, 256, 1, 1, 1, true);
			Init_Sigmoid_Params(cls_loss);
#endif
		}

		Nose_nir_net::~Nose_nir_net()
		{
			delete conv1;
			delete prelu1;
			delete conv1_dw;
			delete prelu1_dw;
			delete conv2;
			delete conv2_dw;
			delete prelu2_dw;
			delete conv3;
			delete conv3_dw;
			delete prelu3_dw;
			delete conv4;
			delete conv4_dw;
			delete prelu4_dw;
			delete conv5;
			delete prelu5;
			delete conv6_1;
			delete cls_loss;

			//conv_weights and bias free automatically, prelu_weights need to free explicitly
			aligned_heap_free(prelu1_weights);
			aligned_heap_free(prelu1_dw_weights);
			aligned_heap_free(prelu2_dw_weights);
			aligned_heap_free(prelu3_dw_weights);
			aligned_heap_free(prelu4_dw_weights);
			aligned_heap_free(prelu5_weights);

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

		void Nose_nir_net::Forward_cpu(const std::shared_ptr<memory::tensor<float>> input_data)
		{
			conv1->Forward(input_data, conv1_top_data);
			prelu1->Forward_cpu(conv1_top_data);
			conv1_dw->Forward(conv1_top_data, conv1_dw_top_data);
			prelu1_dw->Forward_cpu(conv1_dw_top_data);
			conv2->Forward(conv1_dw_top_data, conv2_top_data);
			conv2_dw->Forward(conv2_top_data, conv2_dw_top_data);
			prelu2_dw->Forward_cpu(conv2_dw_top_data);
			conv3->Forward(conv2_dw_top_data, conv3_top_data);
			conv3_dw->Forward(conv3_top_data, conv3_dw_top_data);
			prelu3_dw->Forward_cpu(conv3_dw_top_data);
			conv4->Forward(conv3_dw_top_data, conv4_top_data);
			conv4_dw->Forward(conv4_top_data, conv4_dw_top_data);
			prelu4_dw->Forward_cpu(conv4_dw_top_data);
			conv5->Forward_cpu(conv4_dw_top_data, conv5_top_data);
			prelu5->Forward_cpu(conv5_top_data);
			conv6_1->Forward_cpu(conv5_top_data, conv6_1_top_data);
			cls_loss->Forward_cpu(conv6_1_top_data);
		}

#ifdef USE_CUDA
		void Nose_nir_net::Forward_gpu_native(const std::shared_ptr<memory::tensor<float>> input_data)
		{
			conv1->Forward(cublas_handle_, input_data, conv1_top_data);
			prelu1->Forward_gpu_native(conv1_top_data);
			conv1_dw->Forward(cublas_handle_, conv1_top_data, conv1_dw_top_data);
			prelu1_dw->Forward_gpu_native(conv1_dw_top_data);
			conv2->Forward(cublas_handle_, conv1_dw_top_data, conv2_top_data);
			conv2_dw->Forward(cublas_handle_, conv2_top_data, conv2_dw_top_data);
			prelu2_dw->Forward_gpu_native(conv2_dw_top_data);
			conv3->Forward(cublas_handle_, conv2_dw_top_data, conv3_top_data);
			conv3_dw->Forward(cublas_handle_, conv3_top_data, conv3_dw_top_data);
			prelu3_dw->Forward_gpu_native(conv3_dw_top_data);
			conv4->Forward(cublas_handle_, conv3_dw_top_data, conv4_top_data);
			conv4_dw->Forward(cublas_handle_, conv4_top_data, conv4_dw_top_data);
			prelu4_dw->Forward_gpu_native(conv4_dw_top_data);
			conv5->Forward_gpu_native(cublas_handle_, conv4_dw_top_data, conv5_top_data);
			prelu5->Forward_gpu_native(conv5_top_data);
			conv6_1->Forward_gpu_native(cublas_handle_, conv5_top_data, conv6_1_top_data);
			cls_loss->Forward_gpu_native(conv6_1_top_data);
		}

#ifdef USE_CUDNN
		void Nose_nir_net::Forward_gpu_cudnn(const std::shared_ptr<memory::tensor<float>> input_data)
		{
			conv1->Forward(cudnn_handle_, input_data, conv1_top_data);
			prelu1->Forward_gpu_native(conv1_top_data);
			conv1_dw->Forward(cudnn_handle_, conv1_top_data, conv1_dw_top_data);
			prelu1_dw->Forward_gpu_native(conv1_dw_top_data);
			conv2->Forward(cudnn_handle_, conv1_dw_top_data, conv2_top_data);
			conv2_dw->Forward(cudnn_handle_, conv2_top_data, conv2_dw_top_data);
			prelu2_dw->Forward_gpu_native(conv2_dw_top_data);
			conv3->Forward(cudnn_handle_, conv2_dw_top_data, conv3_top_data);
			conv3_dw->Forward(cudnn_handle_, conv3_top_data, conv3_dw_top_data);
			prelu3_dw->Forward_gpu_native(conv3_dw_top_data);
			conv4->Forward(cudnn_handle_, conv3_dw_top_data, conv4_top_data);
			conv4_dw->Forward(cudnn_handle_, conv4_top_data, conv4_dw_top_data);
			prelu4_dw->Forward_gpu_native(conv4_dw_top_data);
			conv5->Forward_gpu_native(cublas_handle_, conv4_dw_top_data, conv5_top_data);
			prelu5->Forward_gpu_native(conv5_top_data);
			conv6_1->Forward_gpu_native(cublas_handle_, conv5_top_data, conv6_1_top_data);
			cls_loss->Forward_gpu_native(conv6_1_top_data);
		}
#endif 
#endif

	}
}