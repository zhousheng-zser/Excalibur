#include "ipts_net.hpp"

namespace glasssix
{
	ipts_net::ipts_net(int device)
	{
		float quantize_level = INT_MAX;
		Copy_Params(conv1_weights, IPTs_v2, quantize_level);
		Copy_Params(conv1_bias, IPTs_v2, quantize_level);
		Copy_Params(prelu1_weights, IPTs_v2, quantize_level);
		Copy_Params(conv2_weights, IPTs_v2, quantize_level);
		Copy_Params(conv2_bias, IPTs_v2, quantize_level);
		Copy_Params(prelu2_weights, IPTs_v2, quantize_level);
		Copy_Params(conv3_weights, IPTs_v2, quantize_level);
		Copy_Params(conv3_bias, IPTs_v2, quantize_level);
		Copy_Params(prelu3_weights, IPTs_v2, quantize_level);
		Copy_Params(conv4_weights, IPTs_v2, quantize_level);
		Copy_Params(conv4_bias, IPTs_v2, quantize_level);
		Copy_Params(prelu4_weights, IPTs_v2, quantize_level);
		Copy_Params(fc2_weights, IPTs_v2, quantize_level);
		Copy_Params(fc2_bias, IPTs_v2, quantize_level);
		//
		device_ = device;
//#ifdef USE_CUDA
//		if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
//			LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
//		}
//#endif
		//
		Init_Conv_Params(conv1, 3, 16, 5, 1, 0, true);
		Init_PReLU_Params(prelu1, 16, false);
		Init_Pooling_Params(pool1, 2, 2, 0, 0);
		Init_Conv_Params(conv2, 16, 32, 5, 1, 0, true);
		Init_PReLU_Params(prelu2, 32, false);
		Init_Pooling_Params(pool2, 2, 2, 0, 0);
		Init_Conv_Params(conv3, 32, 48, 3, 1, 0, true);
		Init_PReLU_Params(prelu3, 48, false);
		Init_Pooling_Params(pool3, 2, 2, 0, 0);
		Init_Conv_Params(conv4, 48, 64, 3, 1, 0, true);
		Init_PReLU_Params(prelu4, 64, false);
		Init_InnerProduct_Params(fc2, 64, 3, 3, 10, true);
	}


	ipts_net::~ipts_net()
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
		delete fc2;
	}

	void ipts_net::Forward_cpu(const std::shared_ptr<tensor> input_data)
	{
#ifdef _DEBUG
		CHECK_EQ(input_data->width(), 60);
		CHECK_EQ(input_data->height(), 60);
		CHECK_EQ(input_data->channels(), 3);
#endif
		tensor_data.reset(new tensor(input_data->data_shape(), device_));
		float* temp = tensor_data->mutable_cpu_data();
		memcpy(temp, input_data->cpu_data(), input_data->count(0, 4) * sizeof(float));
		conv1->Forward_cpu(tensor_data, conv1_top_data);
		prelu1->Forward_cpu(conv1_top_data);
		pool1->Forward_cpu(conv1_top_data, pool1_top_data);
		conv2->Forward_cpu(pool1_top_data, conv2_top_data);
		prelu2->Forward_cpu(conv2_top_data);
		pool2->Forward_cpu(conv2_top_data, pool2_top_data);
		conv3->Forward_cpu(pool2_top_data, conv3_top_data);
		prelu3->Forward_cpu(conv3_top_data);
		pool3->Forward_cpu(conv3_top_data, pool3_top_data);
		conv4->Forward_cpu(pool3_top_data, conv4_top_data);
		prelu4->Forward_cpu(conv4_top_data);
		fc2->Forward_cpu(conv4_top_data, fc2_top_data);
	}

#ifdef USE_CUDA
	void ipts_net::Forward_native_gpu(const std::shared_ptr<tensor> input_data, cublasHandle_t cublas_handle_)
	{
#ifdef _DEBUG
		CHECK_EQ(input_data->width(), 60);
		CHECK_EQ(input_data->height(), 60);
		CHECK_EQ(input_data->channels(), 3);
#endif
		tensor_data.reset(new tensor(input_data->data_shape(), device_));
		float* temp = tensor_data->mutable_gpu_data();
		math_functions::excalibur_copy(input_data->count(0, 4), input_data->gpu_data(), temp, device_);
		conv1->Forward_native_gpu(cublas_handle_, tensor_data, conv1_top_data);
		prelu1->Forward_native_gpu(conv1_top_data);
		pool1->Forward_native_gpu(conv1_top_data, pool1_top_data);
		conv2->Forward_native_gpu(cublas_handle_, pool1_top_data, conv2_top_data);
		prelu2->Forward_native_gpu(conv2_top_data);
		pool2->Forward_native_gpu(conv2_top_data, pool2_top_data);
		conv3->Forward_native_gpu(cublas_handle_, pool2_top_data, conv3_top_data);
		prelu3->Forward_native_gpu(conv3_top_data);
		pool3->Forward_native_gpu(conv3_top_data, pool3_top_data);
		conv4->Forward_native_gpu(cublas_handle_, pool3_top_data, conv4_top_data);
		prelu4->Forward_native_gpu(conv4_top_data);
		fc2->Forward_native_gpu(cublas_handle_, conv4_top_data, fc2_top_data);
	}

#endif
}
