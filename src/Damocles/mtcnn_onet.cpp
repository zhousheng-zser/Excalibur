#include "mtcnn_onet.hpp"

namespace glasssix
{
	namespace longinus
	{
		mtcnn_onet::mtcnn_onet(int device)
		{
			float quantize_level = INT_MAX;
			Copy_Params(conv1_weights, ONet, quantize_level);
			Copy_Params(conv1_bias, ONet, quantize_level);
			Copy_Params(prelu1_weights, ONet, quantize_level);
			Copy_Params(conv2_weights, ONet, quantize_level);
			Copy_Params(conv2_bias, ONet, quantize_level);
			Copy_Params(prelu2_weights, ONet, quantize_level);
			Copy_Params(conv3_weights, ONet, quantize_level);
			Copy_Params(conv3_bias, ONet, quantize_level);
			Copy_Params(prelu3_weights, ONet, quantize_level);
			Copy_Params(conv4_weights, ONet, quantize_level);
			Copy_Params(conv4_bias, ONet, quantize_level);
			Copy_Params(prelu4_weights, ONet, quantize_level);
			Copy_Params(conv5_weights, ONet, quantize_level);
			Copy_Params(conv5_bias, ONet, quantize_level);
			Copy_Params(prelu5_weights, ONet, quantize_level);
			Copy_Params(conv6_1_weights, ONet, quantize_level);
			Copy_Params(conv6_1_bias, ONet, quantize_level);
			Copy_Params(conv6_2_weights, ONet, quantize_level);
			Copy_Params(conv6_2_bias, ONet, quantize_level);
			Copy_Params(conv6_3_weights, ONet, quantize_level);
			Copy_Params(conv6_3_bias, ONet, quantize_level);
			//
			device_ = device;
#ifdef USE_CUDA
			if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
				LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
			}
#ifdef USE_CUDNN
			/*if (cudnnCreate(&cudnn_handle_) != CUDNN_STATUS_SUCCESS) {
			LOG(ERROR) << "Cannot create Cudnn handle. Cudnn won't be available.";
			}*/
#endif
#endif
			//
			Init_Conv_Params(conv1, 3, 16, 3, 1, 0, true);
			Init_PReLU_Params(prelu1, 16, false);
			Init_Pooling_Params(pool1, 3, 2, 0, 0);
			Init_Conv_Params(conv2, 16, 32, 3, 1, 0, true);
			Init_PReLU_Params(prelu2, 32, false);
			Init_Pooling_Params(pool2, 3, 2, 0, 0);
			Init_Conv_Params(conv3, 32, 32, 3, 1, 0, true);
			Init_PReLU_Params(prelu3, 32, false);
			Init_Pooling_Params(pool3, 2, 2, 0, 0);
			Init_Conv_Params(conv4, 32, 64, 2, 1, 0, true);
			Init_PReLU_Params(prelu4, 64, false);
			Init_InnerProduct_Params(conv5, 64, 3, 3, 128, true);
			Init_PReLU_Params(prelu5, 128, false);
			Init_InnerProduct_Params(conv6_1, 128, 1, 1, 2, true);
			Init_InnerProduct_Params(conv6_2, 128, 1, 1, 4, true);
			Init_InnerProduct_Params(conv6_3, 128, 1, 1, 10, true);
			Init_Softmax_Params(prob1, 2);

			/*Init_Conv_Params(conv1, 3, 32, 3, 1, 0, true);
			Init_PReLU_Params(prelu1, 32, false);
			Init_Pooling_Params(pool1, 3, 2, 0, 0);
			Init_Conv_Params(conv2, 32, 64, 3, 1, 0, true);
			Init_PReLU_Params(prelu2, 64, false);
			Init_Pooling_Params(pool2, 3, 2, 0, 0);
			Init_Conv_Params(conv3, 64, 64, 3, 1, 0, true);
			Init_PReLU_Params(prelu3, 64, false);
			Init_Pooling_Params(pool3, 2, 2, 0, 0);
			Init_Conv_Params(conv4, 64, 128, 2, 1, 0, true);
			Init_PReLU_Params(prelu4, 128, false);
			Init_InnerProduct_Params(conv5, 128, 3, 3, 256, true);
			Init_PReLU_Params(prelu5, 128, false);
			Init_InnerProduct_Params(conv6_1, 256, 1, 1, 2, true);
			Init_InnerProduct_Params(conv6_2, 256, 1, 1, 4, true);
			Init_InnerProduct_Params(conv6_3, 256, 1, 1, 10, true);
			Init_Softmax_Params(prob1, 2);*/
		}


		mtcnn_onet::~mtcnn_onet()
		{
			delete conv1;
			delete prelu1;
			delete pool1;
			delete conv2;
			delete prelu2;
			delete pool2;
			delete conv3;
			delete prelu3;
			delete pool3;
			delete conv4;
			delete prelu4;
			delete conv5;
			delete prelu5;
			delete conv6_1;
			delete conv6_2;
			delete conv6_3;
			delete prob1;
#ifdef USE_CUDA
			if (cublas_handle_)
			{
				CUBLAS_CHECK(cublasDestroy(cublas_handle_));
			}
#endif
		}

		void mtcnn_onet::Forward_cpu(const std::shared_ptr<tensor<float>> input_data)
		{
			conv1->Forward(input_data, conv1_top_data);
			prelu1->Forward_cpu(conv1_top_data);
			pool1->Forward_cpu(conv1_top_data, pool1_top_data);
			conv2->Forward(pool1_top_data, conv2_top_data);
			prelu2->Forward_cpu(conv2_top_data);
			pool2->Forward_cpu(conv2_top_data, pool2_top_data);
			conv3->Forward(pool2_top_data, conv3_top_data);
			prelu3->Forward_cpu(conv3_top_data);
			pool3->Forward_cpu(conv3_top_data, pool3_top_data);
			conv4->Forward(pool3_top_data, conv4_top_data);
			prelu4->Forward_cpu(conv4_top_data);
			conv5->Forward_cpu(conv4_top_data, conv5_top_data);
			prelu5->Forward_cpu(conv5_top_data);
			conv6_1->Forward_cpu(conv5_top_data, conv6_1_top_data);
			conv6_2->Forward_cpu(conv5_top_data, conv6_2_top_data);
			conv6_3->Forward_cpu(conv5_top_data, conv6_3_top_data);
			prob1->Forward_cpu(conv6_1_top_data, prob1_top_data);
		}

#ifdef USE_CUDA
		void mtcnn_onet::Forward_gpu_native(const std::shared_ptr<tensor<float>> input_data)
		{
			conv1->Forward(input_data, conv1_top_data);
			prelu1->Forward_gpu_native(conv1_top_data);
			pool1->Forward_gpu_native(conv1_top_data, pool1_top_data);
			conv2->Forward(pool1_top_data, conv2_top_data);
			prelu2->Forward_gpu_native(conv2_top_data);
			pool2->Forward_gpu_native(conv2_top_data, pool2_top_data);
			conv3->Forward(pool2_top_data, conv3_top_data);
			prelu3->Forward_gpu_native(conv3_top_data);
			pool3->Forward_gpu_native(conv3_top_data, pool3_top_data);
			conv4->Forward(pool3_top_data, conv4_top_data);
			prelu4->Forward_gpu_native(conv4_top_data);
			conv5->Forward_gpu_native(cublas_handle_, conv4_top_data, conv5_top_data);
			prelu5->Forward_gpu_native(conv5_top_data);
			conv6_1->Forward_gpu_native(cublas_handle_, conv5_top_data, conv6_1_top_data);
			conv6_2->Forward_gpu_native(cublas_handle_, conv5_top_data, conv6_2_top_data);
			conv6_3->Forward_gpu_native(cublas_handle_, conv5_top_data, conv6_3_top_data);
			prob1->Forward_gpu_native(conv6_1_top_data, prob1_top_data);
		}
#ifdef USE_CUDNN
		void mtcnn_onet::Forward_gpu_cudnn(const std::shared_ptr<tensor<float>> input_data)
		{
			conv1->Forward(input_data, conv1_top_data);
			prelu1->Forward_gpu_native(conv1_top_data);
			pool1->Forward_gpu_cudnn(conv1_top_data, pool1_top_data);
			conv2->Forward(pool1_top_data, conv2_top_data);
			prelu2->Forward_gpu_native(conv2_top_data);
			pool2->Forward_gpu_cudnn(conv2_top_data, pool2_top_data);
			conv3->Forward(pool2_top_data, conv3_top_data);
			prelu3->Forward_gpu_native(conv3_top_data);
			pool3->Forward_gpu_cudnn(conv3_top_data, pool3_top_data);
			conv4->Forward(pool3_top_data, conv4_top_data);
			prelu4->Forward_gpu_native(conv4_top_data);
			conv5->Forward_gpu_native(cublas_handle_, conv4_top_data, conv5_top_data);
			prelu5->Forward_gpu_native(conv5_top_data);
			conv6_1->Forward_gpu_native(cublas_handle_, conv5_top_data, conv6_1_top_data);
			conv6_2->Forward_gpu_native(cublas_handle_, conv5_top_data, conv6_2_top_data);
			conv6_3->Forward_gpu_native(cublas_handle_, conv5_top_data, conv6_3_top_data);
			prob1->Forward_gpu_cudnn(conv6_1_top_data, prob1_top_data);
		}
#endif
#endif

		void mtcnn_onet::Forward(const std::shared_ptr<tensor<float>> input_data)
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
