#include "mtcnn_rnet.hpp"

namespace glasssix
{
	mtcnn_rnet::mtcnn_rnet(int device)
	{
		
		float quantize_level = INT_MAX;
		Copy_Params(conv1_weights, RNet, quantize_level);
		Copy_Params(conv1_bias, RNet, quantize_level);
		Copy_Params(prelu1_weights, RNet, quantize_level);
		Copy_Params(conv2_weights, RNet, quantize_level);
		Copy_Params(conv2_bias, RNet, quantize_level);
		Copy_Params(prelu2_weights, RNet, quantize_level);
		Copy_Params(conv3_weights, RNet, quantize_level);
		Copy_Params(conv3_bias, RNet, quantize_level);
		Copy_Params(prelu3_weights, RNet, quantize_level);
		Copy_Params(conv4_weights, RNet, quantize_level);
		Copy_Params(conv4_bias, RNet, quantize_level);
		Copy_Params(prelu4_weights, RNet, quantize_level);
		Copy_Params(conv5_1_weights, RNet, quantize_level);
		Copy_Params(conv5_1_bias, RNet, quantize_level);
		Copy_Params(conv5_2_weights, RNet, quantize_level);
		Copy_Params(conv5_2_bias, RNet, quantize_level);
		/*Copy_Params(conv5_3_weights, RNet, quantize_level);
		Copy_Params(conv5_3_bias, RNet, quantize_level);*/
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
		Init_Conv_Params(conv1, 3, 28, 3, 1, 0, true);
		Init_PReLU_Params(prelu1, 28, false);
		Init_Pooling_Params(pool1, 3, 2, 0, 0);
		Init_Conv_Params(conv2, 28, 48, 3, 1, 0, true);
		Init_PReLU_Params(prelu2, 48, false);
		Init_Pooling_Params(pool2, 3, 2, 0, 0);
		Init_Conv_Params(conv3, 48, 64, 2, 1, 0, true);
		Init_PReLU_Params(prelu3, 64, false);
		Init_InnerProduct_Params(conv4, 64, 3, 3, 128, true);
		Init_PReLU_Params(prelu4, 128, false);
		Init_InnerProduct_Params(conv5_1, 128, 1, 1, 2, true);
		Init_InnerProduct_Params(conv5_2, 128, 1, 1, 4, true);
		Init_Softmax_Params(prob1, 2);
	}


	mtcnn_rnet::~mtcnn_rnet()
	{
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
#ifdef USE_CUDA
		if (cublas_handle_)
		{
			CUBLAS_CHECK(cublasDestroy(cublas_handle_));
		}
#ifdef USE_CUDNN
		/*if (cudnn_handle_)
		{
			CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
		}*/
#endif
#endif
	}

	void mtcnn_rnet::Forward_cpu(const std::shared_ptr<tensor<float>> input_data)
	{
		conv1->Forward_cpu(input_data, conv1_top_data);
		prelu1->Forward_cpu(conv1_top_data);
		pool1->Forward_cpu(conv1_top_data, pool1_top_data);
		conv2->Forward_cpu(pool1_top_data, conv2_top_data);
		prelu2->Forward_cpu(conv2_top_data);
		pool2->Forward_cpu(conv2_top_data, pool2_top_data);
		conv3->Forward_cpu(pool2_top_data, conv3_top_data);
		prelu3->Forward_cpu(conv3_top_data);
		conv4->Forward_cpu(conv3_top_data, conv4_top_data);
		prelu4->Forward_cpu(conv4_top_data);
		conv5_1->Forward_cpu(conv4_top_data, conv5_1_top_data);
		conv5_2->Forward_cpu(conv4_top_data, conv5_2_top_data);
		prob1->Forward_cpu(conv5_1_top_data, prob1_top_data);
	}

#ifdef USE_CUDA
	void mtcnn_rnet::Forward_native_gpu(const std::shared_ptr<tensor<float>> input_data)
	{
		conv1->Forward_native_gpu(cublas_handle_, input_data, conv1_top_data);
		prelu1->Forward_native_gpu(conv1_top_data);
		pool1->Forward_native_gpu(conv1_top_data, pool1_top_data);
		conv2->Forward_native_gpu(cublas_handle_, pool1_top_data, conv2_top_data);
		prelu2->Forward_native_gpu(conv2_top_data);
		pool2->Forward_native_gpu(conv2_top_data, pool2_top_data);
		conv3->Forward_native_gpu(cublas_handle_, pool2_top_data, conv3_top_data);
		prelu3->Forward_native_gpu(conv3_top_data);
		conv4->Forward_native_gpu(cublas_handle_, conv3_top_data, conv4_top_data);
		prelu4->Forward_native_gpu(conv4_top_data);
		conv5_1->Forward_native_gpu(cublas_handle_, conv4_top_data, conv5_1_top_data);
		conv5_2->Forward_native_gpu(cublas_handle_, conv4_top_data, conv5_2_top_data);
		prob1->Forward_native_gpu(conv5_1_top_data, prob1_top_data);
	}
#ifdef USE_CUDNN
	void mtcnn_rnet::Forward_cudnn_gpu(const std::shared_ptr<tensor<float>> input_data)
	{
		conv1->Forward_cudnn_gpu(input_data, conv1_top_data);
		prelu1->Forward_native_gpu(conv1_top_data);
		pool1->Forward_cudnn_gpu(conv1_top_data, pool1_top_data);
		conv2->Forward_cudnn_gpu(pool1_top_data, conv2_top_data);
		prelu2->Forward_native_gpu(conv2_top_data);
		pool2->Forward_cudnn_gpu(conv2_top_data, pool2_top_data);
		conv3->Forward_cudnn_gpu(pool2_top_data, conv3_top_data);
		prelu3->Forward_native_gpu(conv3_top_data);
		conv4->Forward_native_gpu(cublas_handle_, conv3_top_data, conv4_top_data);
		prelu4->Forward_native_gpu(conv4_top_data);
		conv5_1->Forward_native_gpu(cublas_handle_, conv4_top_data, conv5_1_top_data);
		conv5_2->Forward_native_gpu(cublas_handle_, conv4_top_data, conv5_2_top_data);
		prob1->Forward_cudnn_gpu(conv5_1_top_data, prob1_top_data);
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
			Forward_cudnn_gpu(input_data);
			return;
#endif
			Forward_native_gpu(input_data);
			return;
#else
			NO_GPU;
#endif
		}
	}

}
