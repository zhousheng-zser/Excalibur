#include "ipbbox_net.hpp"

namespace glasssix
{
	ipbbox_net::ipbbox_net(int device)
	{
		float quantize_level = INT_MAX;
		Copy_Params(conv1_weights, IPBBox_v2, quantize_level);
		Copy_Params(conv1_bias, IPBBox_v2, quantize_level);
		Copy_Params(prelu1_weights, IPBBox_v2, quantize_level);
		Copy_Params(conv2_weights, IPBBox_v2, quantize_level);
		Copy_Params(conv2_bias, IPBBox_v2, quantize_level);
		Copy_Params(prelu2_weights, IPBBox_v2, quantize_level);
		Copy_Params(conv3_weights, IPBBox_v2, quantize_level);
		Copy_Params(conv3_bias, IPBBox_v2, quantize_level);
		Copy_Params(prelu3_weights, IPBBox_v2, quantize_level);
		Copy_Params(conv4_weights, IPBBox_v2, quantize_level);
		Copy_Params(conv4_bias, IPBBox_v2, quantize_level);
		Copy_Params(prelu4_weights, IPBBox_v2, quantize_level);
		Copy_Params(fc1_weights, IPBBox_v2, quantize_level);
		Copy_Params(fc1_bias, IPBBox_v2, quantize_level);
		Copy_Params(fc2_weights, IPBBox_v2, quantize_level);
		Copy_Params(fc2_bias, IPBBox_v2, quantize_level);
		Copy_Params(fc3_weights, IPBBox_v2, quantize_level);
		Copy_Params(fc3_bias, IPBBox_v2, quantize_level);
		//
		device_ = device;
//#ifdef USE_CUDA
//		if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
//			LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
//		}
//#endif
		//
		Init_Conv_Params(conv1, 3, 10, 5, 1, 0, true);
		Init_PReLU_Params(prelu1, 10, false);
		Init_Pooling_Params(pool1, 2, 2, 0, 0);
		Init_Conv_Params(conv2, 10, 20, 5, 1, 0, true);
		Init_PReLU_Params(prelu2, 20, false);
		Init_Pooling_Params(pool2, 2, 2, 0, 0);
		Init_Conv_Params(conv3, 20, 30, 3, 1, 0, true);
		Init_PReLU_Params(prelu3, 30, false);
		Init_Pooling_Params(pool3, 2, 2, 0, 0);
		Init_Conv_Params(conv4, 30, 40, 3, 1, 0, true);
		Init_PReLU_Params(prelu4, 40, false);
		Init_InnerProduct_Params(fc1, 40, 3, 3, 64, true);
		Init_ReLU_Params(prelu5, 64, true);
		Init_InnerProduct_Params(fc2, 64, 1, 1, 4, true);
		Init_InnerProduct_Params(fc3, 64, 1, 1, 3, true);
	}


	ipbbox_net::~ipbbox_net()
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
		delete fc1;
		delete prelu5;
		delete fc2;
		delete fc3;
//#ifdef USE_CUDA
//		if (cublas_handle_)
//		{
//			CUBLAS_CHECK(cublasDestroy(cublas_handle_));
//		}
//#endif
	}

	void ipbbox_net::Forward_cpu(const std::shared_ptr<tensor<float>> input_data)
	{
#ifdef _DEBUG
		CHECK_EQ(input_data->width(), 60);
		CHECK_EQ(input_data->height(), 60);
		CHECK_EQ(input_data->channels(), 3);
#endif
		conv1->Forward_cpu(input_data, conv1_top_data);
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
		fc1->Forward_cpu(conv4_top_data, fc1_top_data);
		prelu5->Forward_cpu(fc1_top_data);
		fc2->Forward_cpu(fc1_top_data, fc2_top_data);
		fc3->Forward_cpu(fc1_top_data, fc3_top_data);
	}

#ifdef USE_CUDA
	void ipbbox_net::Forward_native_gpu(const std::shared_ptr<tensor<float>> input_data, cublasHandle_t cublas_handle_)
	{
#ifdef _DEBUG
		CHECK_EQ(input_data->width(), 60);
		CHECK_EQ(input_data->height(), 60);
		CHECK_EQ(input_data->channels(), 3);
#endif
		conv1->Forward_native_gpu(cublas_handle_, input_data, conv1_top_data);
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
		fc1->Forward_native_gpu(cublas_handle_, conv4_top_data, fc1_top_data);
		prelu5->Forward_native_gpu(fc1_top_data);
		fc2->Forward_native_gpu(cublas_handle_, fc1_top_data, fc2_top_data);
		fc3->Forward_native_gpu(cublas_handle_, fc1_top_data, fc3_top_data);
	}

#endif

}
