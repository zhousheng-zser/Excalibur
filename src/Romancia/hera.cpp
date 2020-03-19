#include "hera.hpp"
#include "hera_data.hpp"
#include <glasssix/tensor.hpp>
#include <glasssix/syncedmem.hpp>
#include <numeric>

using namespace glasssix::excalibur;

namespace glasssix
{
	namespace longinus
	{
		Hera::Hera(int device)
		{
			device_ = device;
			if (device_ >= 0)
			{
				int8_quantization_ = false;//do not use int8 in GPU mode
			}

#if SIMD_TYPE >= SIMDTYPE_SSE
			std::shared_ptr<tensor<float>> bottom_round_ = std::make_shared<tensor<float>>(std::vector<int>{mm_align_size});
			float* bottom_round_data_ = bottom_round_->mutable_cpu_data();
#endif // SIMD_TYPE >= SIMDTYPE_SSE

			float quantize_level = INT_MAX;

			Copy_Params(conv1_weights, Hera, quantize_level);
			Copy_Params(conv1_bias, Hera, quantize_level);
			Copy_Params(convolution1_weights, Hera, quantize_level);
			Copy_Params(convolution1_bias, Hera, quantize_level);
			Copy_Params(convolutiondepthwise1_weights, Hera, quantize_level);
			Copy_Params(convolutiondepthwise1_bias, Hera, quantize_level);
			Copy_Params(convolution2_weights, Hera, quantize_level);
			Copy_Params(convolution2_bias, Hera, quantize_level);
			Copy_Params(convolution3_weights, Hera, quantize_level);
			Copy_Params(convolution3_bias, Hera, quantize_level);
			Copy_Params(convolutiondepthwise2_weights, Hera, quantize_level);
			Copy_Params(convolutiondepthwise2_bias, Hera, quantize_level);
			Copy_Params(convolution4_weights, Hera, quantize_level);
			Copy_Params(convolution4_bias, Hera, quantize_level);
			Copy_Params(convolution5_weights, Hera, quantize_level);
			Copy_Params(convolution5_bias, Hera, quantize_level);
			Copy_Params(convolutiondepthwise3_weights, Hera, quantize_level);
			Copy_Params(convolutiondepthwise3_bias, Hera, quantize_level);
			Copy_Params(convolution6_weights, Hera, quantize_level);
			Copy_Params(convolution6_bias, Hera, quantize_level);
			Copy_Params(convolution7_weights, Hera, quantize_level);
			Copy_Params(convolution7_bias, Hera, quantize_level);
			Copy_Params(convolutiondepthwise4_weights, Hera, quantize_level);
			Copy_Params(convolutiondepthwise4_bias, Hera, quantize_level);
			Copy_Params(convolution8_weights, Hera, quantize_level);
			Copy_Params(convolution8_bias, Hera, quantize_level);
			Copy_Params(convolution9_weights, Hera, quantize_level);
			Copy_Params(convolution9_bias, Hera, quantize_level);
			Copy_Params(convolutiondepthwise5_weights, Hera, quantize_level);
			Copy_Params(convolutiondepthwise5_bias, Hera, quantize_level);
			Copy_Params(convolution10_weights, Hera, quantize_level);
			Copy_Params(convolution10_bias, Hera, quantize_level);
			Copy_Params(convolution11_weights, Hera, quantize_level);
			Copy_Params(convolution11_bias, Hera, quantize_level);
			Copy_Params(convolutiondepthwise6_weights, Hera, quantize_level);
			Copy_Params(convolutiondepthwise6_bias, Hera, quantize_level);
			Copy_Params(convolution12_weights, Hera, quantize_level);
			Copy_Params(convolution12_bias, Hera, quantize_level);
			Copy_Params(fc1_weights, Hera, quantize_level);
			Copy_Params(fc1_bias, Hera, quantize_level);
			Copy_Params(fc2_weights, Hera, quantize_level);
			Copy_Params(fc2_bias, Hera, quantize_level);
			Copy_Params(fc3_weights, Hera, quantize_level);
			Copy_Params(fc3_bias, Hera, quantize_level);
			Copy_Params(fc4_weights, Hera, quantize_level);
			Copy_Params(fc4_bias, Hera, quantize_level);


			//			
#ifdef USE_CUDA
			CUBLAS_CHECK(cublasCreate(&cublas_handle_));
#ifdef USE_CUDNN
			CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
			cudnn_ready_ = true;
#endif
#endif

#ifdef __ARM_NEON
			Init_Conv_arm_Params(conv1, 3, 16, 1, 3, 2, 1, true);//1*3*64*64->1*16*32*32
			Init_ReLU_arm_Params(relu1, 16, true);//1*16*32*32->1*16*32*32
			Init_Conv_arm_Params(convolution1, 16, 96, 1, 1, 1, 0, true);//1*16*32*32->1*96*32*32
			Init_ReLU_arm_Params(relu1_1, 96, true);//1*96*32*32->1*96*32*32
			Init_Conv_arm_Params(convolutiondepthwise1, 96, 96, 96, 3, 2, 1, true);//1*96*32*32->1*96*16*16
			Init_ReLU_arm_Params(relu2, 96, true);//1*96*16*16->1*96*16*16
			Init_Conv_arm_Params(convolution2, 96, 24, 1, 1, 1, 0, true);//1*96*16*16->1*24*16*16
			Init_Conv_arm_Params(convolution3, 24, 144, 1, 1, 1, 0, true);//1*24*16*16->1*144*16*16
			Init_ReLU_arm_Params(relu3, 144, true);//1*144*16*16->1*144*16*16
			Init_Conv_arm_Params(convolutiondepthwise2, 144, 144, 144, 3, 1, 1, true);//1*144*16*16->1*144*16*16
			Init_ReLU_arm_Params(relu4, 144, true);//1*144*16*16->1*144*16*16
			Init_Conv_arm_Params(convolution4, 144, 24, 1, 1, 1, 0, true);//1*144*16*16->1*24*16*16
			Init_Eltwise_arm_Params(fuse3, 0);//1*24*16*16->1*24*16*16
			Init_Conv_arm_Params(convolution5, 24, 144, 1, 1, 1, 0, true);//1*24*16*16->1*144*16*16
			Init_ReLU_arm_Params(relu5, 144, true);//1*144*16*16->1*144*16*16
			Init_Conv_arm_Params(convolutiondepthwise3, 144, 144, 144, 3, 2, 1, true);//1*144*16*16->1*144*8*8
			Init_ReLU_arm_Params(relu6, 144, true);//1*144*8*8->1*144*8*8
			Init_Conv_arm_Params(convolution6, 144, 32, 1, 1, 1, 0, true);//1*144*8*8->1*32*8*8
			Init_Conv_arm_Params(convolution7, 32, 192, 1, 1, 1, 0, true);//1*32*8*8->1*192*8*8
			Init_ReLU_arm_Params(relu7, 192, true);//1*192*8*8->1*192*8*8
			Init_Conv_arm_Params(convolutiondepthwise4, 192, 192, 192, 3, 1, 1, true);//1*192*8*8->1*192*8*8
			Init_ReLU_arm_Params(relu8, 192, true);//1*192*8*8->1*192*8*8
			Init_Conv_arm_Params(convolution8, 192, 32, 1, 1, 1, 0, true);//1*192*8*8->1*32*8*8
			Init_Eltwise_arm_Params(fuse5, 0);//1*32*8*8->1*32*8*8
			Init_Conv_arm_Params(convolution9, 32, 192, 1, 1, 1, 0, true);//1*32*8*8->1*192*8*8
			Init_ReLU_arm_Params(relu9, 192, true);//1*192*8*8->1*192*8*8
			Init_Conv_arm_Params(convolutiondepthwise5, 192, 192, 192, 3, 2, 1, true);//1*192*8*8->1*192*4*4
			Init_ReLU_arm_Params(relu10, 192, true);//1*192*4*4->1*192*4*4
			Init_Conv_arm_Params(convolution10, 192, 64, 1, 1, 1, 0, true);//1*192*4*4->1*64*4*4
			Init_Conv_arm_Params(convolution11, 64, 192, 1, 1, 1, 0, true);//1*64*4*4->1*192*4*4
			Init_ReLU_arm_Params(relu11, 192, true);//1*192*4*4->1*192*4*4
			Init_Conv_arm_Params(convolutiondepthwise6, 192, 192, 192, 3, 1, 1, true);//1*192*4*4->1*192*4*4
			Init_ReLU_arm_Params(relu12, 192, true);//1*192*4*4->1*192*4*4
			Init_Conv_arm_Params(convolution12, 192, 64, 1, 1, 1, 0, true);//1*192*4*4->1*64*4*4
			Init_Eltwise_arm_Params(fuse7, 0);//1*64*4*4->1*64*4*4
			Init_InnerProduct_arm_Params(fc1, 64, 4, 4, 200, true);//1*64*4*4->1*200
			Init_ReLU_arm_Params(fc1_relu, 200, true);//1*200->1*200
			Init_InnerProduct_arm_Params(fc2, 200, 1, 1, 200, true);//1*200->1*200
			Init_ReLU_arm_Params(fc2_relu, 200, true);//1*200->1*200
			Init_InnerProduct_arm_Params(fc3, 200, 1, 1, 50, true);//1*200->1*50
			Init_InnerProduct_arm_Params(fc4, 50, 1, 1, 136, true);//1*50->1*136
#else
			Init_Conv_Params(conv1, 3, 16, 1, 3, 2, 1, true);//1*3*64*64->1*16*32*32
			Init_ReLU_Params(relu1, 16, true);//1*16*32*32->1*16*32*32
			Init_Conv_Params(convolution1, 16, 96, 1, 1, 1, 0, true);//1*16*32*32->1*96*32*32
			Init_ReLU_Params(relu1_1, 96, true);//1*96*32*32->1*96*32*32
			Init_Conv_Params(convolutiondepthwise1, 96, 96, 96, 3, 2, 1, true);//1*96*32*32->1*96*16*16
			Init_ReLU_Params(relu2, 96, true);//1*96*16*16->1*96*16*16
			Init_Conv_Params(convolution2, 96, 24, 1, 1, 1, 0, true);//1*96*16*16->1*24*16*16
			Init_Conv_Params(convolution3, 24, 144, 1, 1, 1, 0, true);//1*24*16*16->1*144*16*16
			Init_ReLU_Params(relu3, 144, true);//1*144*16*16->1*144*16*16
			Init_Conv_Params(convolutiondepthwise2, 144, 144, 144, 3, 1, 1, true);//1*144*16*16->1*144*16*16
			Init_ReLU_Params(relu4, 144, true);//1*144*16*16->1*144*16*16
			Init_Conv_Params(convolution4, 144, 24, 1, 1, 1, 0, true);//1*144*16*16->1*24*16*16
			Init_Eltwise_Params(fuse3, 0);//1*24*16*16->1*24*16*16
			Init_Conv_Params(convolution5, 24, 144, 1, 1, 1, 0, true);//1*24*16*16->1*144*16*16
			Init_ReLU_Params(relu5, 144, true);//1*144*16*16->1*144*16*16
			Init_Conv_Params(convolutiondepthwise3, 144, 144, 144, 3, 2, 1, true);//1*144*16*16->1*144*8*8
			Init_ReLU_Params(relu6, 144, true);//1*144*8*8->1*144*8*8
			Init_Conv_Params(convolution6, 144, 32, 1, 1, 1, 0, true);//1*144*8*8->1*32*8*8
			Init_Conv_Params(convolution7, 32, 192, 1, 1, 1, 0, true);//1*32*8*8->1*192*8*8
			Init_ReLU_Params(relu7, 192, true);//1*192*8*8->1*192*8*8
			Init_Conv_Params(convolutiondepthwise4, 192, 192, 192, 3, 1, 1, true);//1*192*8*8->1*192*8*8
			Init_ReLU_Params(relu8, 192, true);//1*192*8*8->1*192*8*8
			Init_Conv_Params(convolution8, 192, 32, 1, 1, 1, 0, true);//1*192*8*8->1*32*8*8
			Init_Eltwise_Params(fuse5, 0);//1*32*8*8->1*32*8*8
			Init_Conv_Params(convolution9, 32, 192, 1, 1, 1, 0, true);//1*32*8*8->1*192*8*8
			Init_ReLU_Params(relu9, 192, true);//1*192*8*8->1*192*8*8
			Init_Conv_Params(convolutiondepthwise5, 192, 192, 192, 3, 2, 1, true);//1*192*8*8->1*192*4*4
			Init_ReLU_Params(relu10, 192, true);//1*192*4*4->1*192*4*4
			Init_Conv_Params(convolution10, 192, 64, 1, 1, 1, 0, true);//1*192*4*4->1*64*4*4
			Init_Conv_Params(convolution11, 64, 192, 1, 1, 1, 0, true);//1*64*4*4->1*192*4*4
			Init_ReLU_Params(relu11, 192, true);//1*192*4*4->1*192*4*4
			Init_Conv_Params(convolutiondepthwise6, 192, 192, 192, 3, 1, 1, true);//1*192*4*4->1*192*4*4
			Init_ReLU_Params(relu12, 192, true);//1*192*4*4->1*192*4*4
			Init_Conv_Params(convolution12, 192, 64, 1, 1, 1, 0, true);//1*192*4*4->1*64*4*4
			Init_Eltwise_Params(fuse7, 0);//1*64*4*4->1*64*4*4
			Init_InnerProduct_Params(fc1, 64, 4, 4, 200, true);//1*64*4*4->1*200
			Init_ReLU_Params(fc1_relu, 200, true);//1*200->1*200
			Init_InnerProduct_Params(fc2, 200, 1, 1, 200, true);//1*200->1*200
			Init_ReLU_Params(fc2_relu, 200, true);//1*200->1*200
			Init_InnerProduct_Params(fc3, 200, 1, 1, 50, true);//1*200->1*50
			Init_InnerProduct_Params(fc4, 50, 1, 1, 136, true);//1*50->1*136
#endif
		}


		Hera::~Hera()
		{
			delete conv1;
			delete relu1;
			delete convolution1;
			delete relu1;
			delete convolutiondepthwise1;
			delete relu2;
			delete convolution2;
			delete convolution3;
			delete relu3;
			delete convolutiondepthwise2;
			delete relu4;
			delete convolution4;
			delete fuse3;
			delete convolution5;
			delete relu5;
			delete convolutiondepthwise3;
			delete relu6;
			delete convolution6;
			delete convolution7;
			delete relu7;
			delete convolutiondepthwise4;
			delete relu8;
			delete convolution8;
			delete fuse5;
			delete convolution9;
			delete relu9;
			delete convolutiondepthwise5;
			delete relu10;
			delete convolution10;
			delete convolution11;
			delete relu11;
			delete convolutiondepthwise6;
			delete relu12;
			delete convolution12;
			delete fuse7;
			delete fc1;
			delete fc1_relu;
			delete fc2;
			delete fc2_relu;
			delete fc3;
			delete fc4;

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

		void Hera::Forward_cpu(const std::shared_ptr<tensor<float>> input_data)
		{
			conv1->Forward(input_data, conv1_top_data);
			relu1->Forward_cpu(conv1_top_data);
			convolution1->Forward(conv1_top_data, convolution1_top_data);
			relu1->Forward_cpu(convolution1_top_data);
			convolutiondepthwise1->Forward(convolution1_top_data, convolutiondepthwise1_top_data);
			relu2->Forward_cpu(convolutiondepthwise1_top_data);
			convolution2->Forward(convolutiondepthwise1_top_data, convolution2_top_data);
			convolution3->Forward(convolution2_top_data, convolution3_top_data);
			relu3->Forward_cpu(convolution3_top_data);
			convolutiondepthwise2->Forward(convolution3_top_data, convolutiondepthwise2_top_data);
			relu4->Forward_cpu(convolutiondepthwise2_top_data);
			convolution4->Forward(convolutiondepthwise2_top_data, convolution4_top_data);
			fuse3->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{convolution2_top_data, convolution4_top_data}, fuse3_top_data);
			convolution5->Forward(fuse3_top_data, convolution5_top_data);
			relu5->Forward_cpu(convolution5_top_data);
			convolutiondepthwise3->Forward(convolution5_top_data, convolutiondepthwise3_top_data);
			relu6->Forward_cpu(convolutiondepthwise3_top_data);
			convolution6->Forward(convolutiondepthwise3_top_data, convolution6_top_data);
			convolution7->Forward(convolution6_top_data, convolution7_top_data);
			relu7->Forward_cpu(convolution7_top_data);
			convolutiondepthwise4->Forward(convolution7_top_data, convolutiondepthwise4_top_data);
			relu8->Forward_cpu(convolutiondepthwise4_top_data);
			convolution8->Forward(convolutiondepthwise4_top_data, convolution8_top_data);
			fuse5->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{convolution6_top_data, convolution8_top_data}, fuse5_top_data);
			convolution9->Forward(fuse5_top_data, convolution9_top_data);
			relu9->Forward_cpu(convolution9_top_data);
			convolutiondepthwise5->Forward(convolution9_top_data, convolutiondepthwise5_top_data);
			relu10->Forward_cpu(convolutiondepthwise5_top_data);
			convolution10->Forward(convolutiondepthwise5_top_data, convolution10_top_data);
			convolution11->Forward(convolution10_top_data, convolution11_top_data);
			relu11->Forward_cpu(convolution11_top_data);
			convolutiondepthwise6->Forward(convolution11_top_data, convolutiondepthwise6_top_data);
			relu12->Forward_cpu(convolutiondepthwise6_top_data);
			convolution12->Forward(convolutiondepthwise6_top_data, convolution12_top_data);
			fuse7->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{convolution10_top_data, convolution12_top_data}, fuse7_top_data);
			fc1->Forward_cpu(fuse7_top_data, fc1_top_data);
			fc1_relu->Forward_cpu(fc1_top_data);
			fc2->Forward_cpu(fc1_top_data, fc2_top_data);
			fc2_relu->Forward_cpu(fc2_top_data);
			fc3->Forward_cpu(fc2_top_data, fc3_top_data);
			fc4->Forward_cpu(fc3_top_data, fc4_top_data);
		}

#ifdef USE_CUDA

#ifdef USE_CUDNN
		void Hera::Forward_gpu_cudnn(const std::shared_ptr<tensor<float>> input_data)
		{
			conv1->Forward(cudnn_handle_, input_data, conv1_top_data);
			relu1->Forward_gpu_native(conv1_top_data);
			convolution1->Forward(cudnn_handle_, conv1_top_data, convolution1_top_data);
			relu1_1->Forward_gpu_native(convolution1_top_data);
			convolutiondepthwise1->Forward(cudnn_handle_, convolution1_top_data, convolutiondepthwise1_top_data);
			relu2->Forward_gpu_native(convolutiondepthwise1_top_data);
			convolution2->Forward(cudnn_handle_, convolutiondepthwise1_top_data, convolution2_top_data);
			convolution3->Forward(cudnn_handle_, convolution2_top_data, convolution3_top_data);
			relu3->Forward_gpu_native(convolution3_top_data);
			convolutiondepthwise2->Forward(cudnn_handle_, convolution3_top_data, convolutiondepthwise2_top_data);
			relu4->Forward_gpu_native(convolutiondepthwise2_top_data);
			convolution4->Forward(cudnn_handle_, convolutiondepthwise2_top_data, convolution4_top_data);
			fuse3->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{convolution2_top_data, convolution4_top_data}, fuse3_top_data);
			convolution5->Forward(cudnn_handle_, fuse3_top_data, convolution5_top_data);
			relu5->Forward_gpu_native(convolution5_top_data);
			convolutiondepthwise3->Forward(cudnn_handle_, convolution5_top_data, convolutiondepthwise3_top_data);
			relu6->Forward_gpu_native(convolutiondepthwise3_top_data);
			convolution6->Forward(cudnn_handle_, convolutiondepthwise3_top_data, convolution6_top_data);
			convolution7->Forward(cudnn_handle_, convolution6_top_data, convolution7_top_data);
			relu7->Forward_gpu_native(convolution7_top_data);
			convolutiondepthwise4->Forward(cudnn_handle_, convolution7_top_data, convolutiondepthwise4_top_data);
			relu8->Forward_gpu_native(convolutiondepthwise4_top_data);
			convolution8->Forward(cudnn_handle_, convolutiondepthwise4_top_data, convolution8_top_data);
			fuse5->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{convolution6_top_data, convolution8_top_data}, fuse5_top_data);
			convolution9->Forward(cudnn_handle_, fuse5_top_data, convolution9_top_data);
			relu9->Forward_gpu_native(convolution9_top_data);
			convolutiondepthwise5->Forward(cudnn_handle_, convolution9_top_data, convolutiondepthwise5_top_data);
			relu10->Forward_gpu_native(convolutiondepthwise5_top_data);
			convolution10->Forward(cudnn_handle_, convolutiondepthwise5_top_data, convolution10_top_data);
			convolution11->Forward(cudnn_handle_, convolution10_top_data, convolution11_top_data);
			relu11->Forward_gpu_native(convolution11_top_data);
			convolutiondepthwise6->Forward(cudnn_handle_, convolution11_top_data, convolutiondepthwise6_top_data);
			relu12->Forward_gpu_native(convolutiondepthwise6_top_data);
			convolution12->Forward(cudnn_handle_, convolutiondepthwise6_top_data, convolution12_top_data);
			fuse7->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{convolution10_top_data, convolution12_top_data}, fuse7_top_data);
			fc1->Forward_gpu_native(cublas_handle_, fuse7_top_data, fc1_top_data);
			fc1_relu->Forward_gpu_native(fc1_top_data);
			fc2->Forward_gpu_native(cublas_handle_, fc1_top_data, fc2_top_data);
			fc2_relu->Forward_gpu_native(fc2_top_data);
			fc3->Forward_gpu_native(cublas_handle_, fc2_top_data, fc3_top_data);
			fc4->Forward_gpu_native(cublas_handle_, fc3_top_data, fc4_top_data);
		}

#else
		void Hera::Forward_gpu_native(const std::shared_ptr<tensor<float>> input_data)
		{
			conv1->Forward(cublas_handle_, input_data, conv1_top_data);
			relu1->Forward_gpu_native(conv1_top_data);
			convolution1->Forward(cublas_handle_, conv1_top_data, convolution1_top_data);
			relu1_1->Forward_gpu_native(convolution1_top_data);
			convolutiondepthwise1->Forward(cublas_handle_, convolution1_top_data, convolutiondepthwise1_top_data);
			relu2->Forward_gpu_native(convolutiondepthwise1_top_data);
			convolution2->Forward(cublas_handle_, convolutiondepthwise1_top_data, convolution2_top_data);
			convolution3->Forward(cublas_handle_, convolution2_top_data, convolution3_top_data);
			relu3->Forward_gpu_native(convolution3_top_data);
			convolutiondepthwise2->Forward(cublas_handle_, convolution3_top_data, convolutiondepthwise2_top_data);
			relu4->Forward_gpu_native(convolutiondepthwise2_top_data);
			convolution4->Forward(cublas_handle_, convolutiondepthwise2_top_data, convolution4_top_data);
			fuse3->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{convolution2_top_data, convolution4_top_data}, fuse3_top_data);
			convolution5->Forward(cublas_handle_, fuse3_top_data, convolution5_top_data);
			relu5->Forward_gpu_native(convolution5_top_data);
			convolutiondepthwise3->Forward(cublas_handle_, convolution5_top_data, convolutiondepthwise3_top_data);
			relu6->Forward_gpu_native(convolutiondepthwise3_top_data);
			convolution6->Forward(cublas_handle_, convolutiondepthwise3_top_data, convolution6_top_data);
			convolution7->Forward(cublas_handle_, convolution6_top_data, convolution7_top_data);
			relu7->Forward_gpu_native(convolution7_top_data);
			convolutiondepthwise4->Forward(cublas_handle_, convolution7_top_data, convolutiondepthwise4_top_data);
			relu8->Forward_gpu_native(convolutiondepthwise4_top_data);
			convolution8->Forward(cublas_handle_, convolutiondepthwise4_top_data, convolution8_top_data);
			fuse5->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{convolution6_top_data, convolution8_top_data}, fuse5_top_data);
			convolution9->Forward(cublas_handle_, fuse5_top_data, convolution9_top_data);
			relu9->Forward_gpu_native(convolution9_top_data);
			convolutiondepthwise5->Forward(cublas_handle_, convolution9_top_data, convolutiondepthwise5_top_data);
			relu10->Forward_gpu_native(convolutiondepthwise5_top_data);
			convolution10->Forward(cublas_handle_, convolutiondepthwise5_top_data, convolution10_top_data);
			convolution11->Forward(cublas_handle_, convolution10_top_data, convolution11_top_data);
			relu11->Forward_gpu_native(convolution11_top_data);
			convolutiondepthwise6->Forward(cublas_handle_, convolution11_top_data, convolutiondepthwise6_top_data);
			relu12->Forward_gpu_native(convolutiondepthwise6_top_data);
			convolution12->Forward(cublas_handle_, convolutiondepthwise6_top_data, convolution12_top_data);
			fuse7->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{convolution10_top_data, convolution12_top_data}, fuse7_top_data);
			fc1->Forward_gpu_native(cublas_handle_, fuse7_top_data, fc1_top_data);
			fc1_relu->Forward_gpu_native(fc1_top_data);
			fc2->Forward_gpu_native(cublas_handle_, fc1_top_data, fc2_top_data);
			fc2_relu->Forward_gpu_native(fc2_top_data);
			fc3->Forward_gpu_native(cublas_handle_, fc2_top_data, fc3_top_data);
			fc4->Forward_gpu_native(cublas_handle_, fc3_top_data, fc4_top_data);
		}
#endif // !USE_CUDNN

#endif // !USE_CUDA

	}
}