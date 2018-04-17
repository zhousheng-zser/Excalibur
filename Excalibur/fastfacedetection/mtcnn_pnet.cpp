#include "mtcnn_pnet.hpp"
#include <filesystem>
#include <iostream>

namespace glasssix
{
	mtcnn_pnet::mtcnn_pnet(int device)
	{
		float quantize_level = INT_MAX;
		Copy_Params(conv1_weights, PNet, quantize_level);
		Copy_Params(conv1_bias, PNet, quantize_level);
		Copy_Params(prelu1_weights, PNet, quantize_level);
		Copy_Params(conv2_weights, PNet, quantize_level);
		Copy_Params(conv2_bias, PNet, quantize_level);
		Copy_Params(prelu2_weights, PNet, quantize_level);
		Copy_Params(conv3_weights, PNet, quantize_level);
		Copy_Params(conv3_bias, PNet, quantize_level);
		Copy_Params(prelu3_weights, PNet, quantize_level);
		Copy_Params(conv4_1_weights, PNet, quantize_level);
		Copy_Params(conv4_1_bias, PNet, quantize_level);
		Copy_Params(conv4_2_weights, PNet, quantize_level);
		Copy_Params(conv4_2_bias, PNet, quantize_level);

		device_ = device;
#ifdef USE_CUDA
		if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
			LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
		}
#ifdef USE_CUDNN
		if (cudnnCreate(&cudnn_handle_) != CUDNN_STATUS_SUCCESS) {
			LOG(ERROR) << "Cannot create Cudnn handle. Cudnn won't be available.";
		}
#endif
#endif
		
		//
		Init_Conv_Params(conv1, 3, 10, 3, 1, 0, true);
		Init_PReLU_Params(prelu1, 10, false);
		Init_Pooling_Params(pool1, 2, 2, 0, 0);
		Init_Conv_Params(conv2, 10, 16, 3, 1, 0, true);
		Init_PReLU_Params(prelu2, 16, false);
		Init_Conv_Params(conv3, 16, 32, 3, 1, 0, true);
		Init_PReLU_Params(prelu3, 32, false);
		Init_Conv_Params(conv4_1, 32, 2, 1, 1, 0, true);
		Init_Conv_Params(conv4_2, 32, 4, 1, 1, 0, true);
		Init_Softmax_Params(prob1, 2);
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

	void mtcnn_pnet::Forward_cpu(const std::shared_ptr<tensor<float>> input_data)
	{		
		tensor_data.reset(new tensor<float>(input_data->data_shape(), -1));
		float* temp = tensor_data->mutable_cpu_data();
		memcpy(temp, input_data->cpu_data(), input_data->count(0, 4) * sizeof(float));
		conv1->Forward_cpu(tensor_data, conv1_top_data);
		prelu1->Forward_cpu(conv1_top_data);
		pool1->Forward_cpu(conv1_top_data, pool1_top_data);
		conv2->Forward_cpu(pool1_top_data, conv2_top_data);
		prelu2->Forward_cpu(conv2_top_data);
		conv3->Forward_cpu(conv2_top_data, conv3_top_data);
		prelu3->Forward_cpu(conv3_top_data);
		conv4_1->Forward_cpu(conv3_top_data, conv4_1_top_data);
		conv4_2->Forward_cpu(conv3_top_data, conv4_2_top_data);
		prob1->Forward_cpu(conv4_1_top_data, prob1_top_data);
	}


#ifdef USE_CUDA
	void mtcnn_pnet::Forward_native_gpu(const std::shared_ptr<tensor<float>> input_data)
	{
		tensor_data.reset(new tensor<float>(input_data->data_shape(), device_));
		float* temp = tensor_data->mutable_gpu_data();
		math_functions::excalibur_copy(input_data->count(0, 4), input_data->gpu_data(), temp, device_);
		conv1->Forward_native_gpu(cublas_handle_, tensor_data, conv1_top_data);
		prelu1->Forward_native_gpu(conv1_top_data);
		pool1->Forward_native_gpu(conv1_top_data, pool1_top_data);
		conv2->Forward_native_gpu(cublas_handle_, pool1_top_data, conv2_top_data);
		prelu2->Forward_native_gpu(conv2_top_data);
		conv3->Forward_native_gpu(cublas_handle_, conv2_top_data, conv3_top_data);
		prelu3->Forward_native_gpu(conv3_top_data);
		conv4_1->Forward_native_gpu(cublas_handle_, conv3_top_data, conv4_1_top_data);
		conv4_2->Forward_native_gpu(cublas_handle_, conv3_top_data, conv4_2_top_data);
		prob1->Forward_native_gpu(conv4_1_top_data, prob1_top_data);
		
	}

#ifdef USE_CUDNN
	void mtcnn_pnet::Forward_cudnn_gpu(const std::shared_ptr<tensor<float>> input_data)
	{
		tensor_data.reset(new tensor<float>(input_data->data_shape(), device_));
		float* temp = tensor_data->mutable_gpu_data();
		math_functions::excalibur_copy(input_data->count(0, 4), input_data->gpu_data(), temp, device_);
		conv1->Forward_cudnn_gpu(cudnn_handle_, tensor_data, conv1_top_data);
		prelu1->Forward_native_gpu(conv1_top_data);
		pool1->Forward_cudnn_gpu(cudnn_handle_, conv1_top_data, pool1_top_data);
		conv2->Forward_cudnn_gpu(cudnn_handle_, pool1_top_data, conv2_top_data);
		prelu2->Forward_native_gpu(conv2_top_data);
		conv3->Forward_cudnn_gpu(cudnn_handle_, conv2_top_data, conv3_top_data);
		prelu3->Forward_native_gpu(conv3_top_data);
		conv4_1->Forward_cudnn_gpu(cudnn_handle_, conv3_top_data, conv4_1_top_data);
		conv4_2->Forward_cudnn_gpu(cudnn_handle_, conv3_top_data, conv4_2_top_data);
		prob1->Forward_native_gpu(conv4_1_top_data, prob1_top_data);
	}

#endif
#endif

	void mtcnn_pnet::Forward(const std::shared_ptr<tensor<float>> input_data)
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
