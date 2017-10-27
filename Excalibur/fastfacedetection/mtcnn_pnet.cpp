#include "mtcnn_pnet.hpp"
#include <filesystem>
#include <iostream>

namespace fastface
{
	mtcnn_pnet::mtcnn_pnet(int device)
	{
		Copy_Params(conv1_weights, PNet);
		Copy_Params(conv1_bias, PNet);
		Copy_Params(prelu1_weights, PNet);
		Copy_Params(conv2_weights, PNet);
		Copy_Params(conv2_bias, PNet);
		Copy_Params(prelu2_weights, PNet);
		Copy_Params(conv3_weights, PNet);
		Copy_Params(conv3_bias, PNet);
		Copy_Params(prelu3_weights, PNet);
		Copy_Params(conv4_1_weights, PNet);
		Copy_Params(conv4_1_bias, PNet);
		Copy_Params(conv4_2_weights, PNet);
		Copy_Params(conv4_2_bias, PNet);

		device_ = device;
#ifdef USE_CUDA
		if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
			LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
		}
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
#endif
	}

	void mtcnn_pnet::Forward_cpu(const std::shared_ptr<tensor> input_data)
	{		
		tensor_data.reset(new tensor(input_data->data_shape(), -1));
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
	void mtcnn_pnet::Forward_native_gpu(const std::shared_ptr<tensor> input_data)
	{
		tensor_data.reset(new tensor(input_data->data_shape(), device_));
		float* temp = tensor_data->mutable_gpu_data();
		math_functions::excalibur_copy(input_data->count(0, 4), input_data->gpu_data(), temp, device_);
		
		conv1->Forward_native_gpu(cublas_handle_, tensor_data, conv1_top_data);
		//auto p0 = std::chrono::system_clock::now();
		prelu1->Forward_native_gpu(conv1_top_data);
		
		pool1->Forward_native_gpu(conv1_top_data, pool1_top_data);
		
		conv2->Forward_native_gpu(cublas_handle_, pool1_top_data, conv2_top_data);
		/*auto p1 = std::chrono::system_clock::now();
		std::cout << "forward conv1 time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << std::endl;*/
		prelu2->Forward_native_gpu(conv2_top_data);
		conv3->Forward_native_gpu(cublas_handle_, conv2_top_data, conv3_top_data);
		prelu3->Forward_native_gpu(conv3_top_data);
		conv4_1->Forward_native_gpu(cublas_handle_, conv3_top_data, conv4_1_top_data);
		conv4_2->Forward_native_gpu(cublas_handle_, conv3_top_data, conv4_2_top_data);
		prob1->Forward_native_gpu(conv4_1_top_data, prob1_top_data);
		
	}

#ifdef USE_CUDNN
	void mtcnn_pnet::Forward_cudnn_gpu(const std::shared_ptr<tensor> input_data)
	{
		/*tensor_data.reset(new tensor(input_data->data_shape(), device_));
		float* temp = tensor_data->mutable_gpu_data();
		math_functions::excalibur_copy(input_data->count(0, 4), input_data->gpu_data(), temp, device_);
		conv1->Forward_cudnn_gpu( tensor_data, conv1_top_data);
		prelu1->Forward_native_gpu(conv1_top_data);
		pool1->Forward_native_gpu(conv1_top_data, pool1_top_data);
		conv2->Forward_cudnn_gpu( pool1_top_data, conv2_top_data);
		prelu2->Forward_native_gpu(conv2_top_data);
		conv3->Forward_cudnn_gpu( conv2_top_data, conv3_top_data);
		prelu3->Forward_native_gpu(conv3_top_data);
		conv4_1->Forward_cudnn_gpu( conv3_top_data, conv4_1_top_data);
		conv4_2->Forward_cudnn_gpu( conv3_top_data, conv4_2_top_data);
		prob1->Forward_native_gpu(conv4_1_top_data, prob1_top_data);*/
	}

#endif

#endif

	void mtcnn_pnet::Forward(const std::shared_ptr<tensor> input_data)
	{
		if (device_<0)
		{
			Forward_cpu(input_data);
		}
		else
		{
#ifdef USE_CUDA
			Forward_native_gpu(input_data);
#else
			NO_GPU;
#endif
		}
	}
}
