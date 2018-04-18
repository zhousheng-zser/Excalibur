#include "unicorn_net.hpp"
#include <iostream>

namespace glasssix
{
	unicorn_net::unicorn_net(int device)
	{
		float quantize_level = USHRT_MAX;
		Copy_Params(conv1a_weights, Unicorn, quantize_level);
		Copy_Params(conv1a_bias, Unicorn, quantize_level);
		Copy_Params(relu1a_weights, Unicorn, quantize_level);
		Copy_Params(conv1b_weights, Unicorn, quantize_level);
		Copy_Params(conv1b_bias, Unicorn, quantize_level);
		Copy_Params(relu1b_weights, Unicorn, quantize_level);
		Copy_Params(conv2_1_weights, Unicorn, quantize_level);
		Copy_Params(conv2_1_bias, Unicorn, quantize_level);
		Copy_Params(relu2_1_weights, Unicorn, quantize_level);
		Copy_Params(conv2_2_weights, Unicorn, quantize_level);
		Copy_Params(conv2_2_bias, Unicorn, quantize_level);
		Copy_Params(relu2_2_weights, Unicorn, quantize_level);
		Copy_Params(conv2_weights, Unicorn, quantize_level);
		Copy_Params(conv2_bias, Unicorn, quantize_level);
		Copy_Params(relu2_weights, Unicorn, quantize_level);
		Copy_Params(conv3_1_weights, Unicorn, quantize_level);
		Copy_Params(conv3_1_bias, Unicorn, quantize_level);
		Copy_Params(relu3_1_weights, Unicorn, quantize_level);
		Copy_Params(conv3_2_weights, Unicorn, quantize_level);
		Copy_Params(conv3_2_bias, Unicorn, quantize_level);
		Copy_Params(relu3_2_weights, Unicorn, quantize_level);
		Copy_Params(conv3_3_weights, Unicorn, quantize_level);
		Copy_Params(conv3_3_bias, Unicorn, quantize_level);
		Copy_Params(relu3_3_weights, Unicorn, quantize_level);
		Copy_Params(conv3_4_weights, Unicorn, quantize_level);
		Copy_Params(conv3_4_bias, Unicorn, quantize_level);
		Copy_Params(relu3_4_weights, Unicorn, quantize_level);
		Copy_Params(conv3_weights, Unicorn, quantize_level);
		Copy_Params(conv3_bias, Unicorn, quantize_level);
		Copy_Params(relu3_weights, Unicorn, quantize_level);
		Copy_Params(conv4_1_weights, Unicorn, quantize_level);
		Copy_Params(conv4_1_bias, Unicorn, quantize_level);
		Copy_Params(relu4_1_weights, Unicorn, quantize_level);
		Copy_Params(conv4_2_weights, Unicorn, quantize_level);
		Copy_Params(conv4_2_bias, Unicorn, quantize_level);
		Copy_Params(relu4_2_weights, Unicorn, quantize_level);
		Copy_Params(conv4_3_weights, Unicorn, quantize_level);
		Copy_Params(conv4_3_bias, Unicorn, quantize_level);
		Copy_Params(relu4_3_weights, Unicorn, quantize_level);
		Copy_Params(conv4_4_weights, Unicorn, quantize_level);
		Copy_Params(conv4_4_bias, Unicorn, quantize_level);
		Copy_Params(relu4_4_weights, Unicorn, quantize_level);
		Copy_Params(conv4_5_weights, Unicorn, quantize_level);
		Copy_Params(conv4_5_bias, Unicorn, quantize_level);
		Copy_Params(relu4_5_weights, Unicorn, quantize_level);
		Copy_Params(conv4_6_weights, Unicorn, quantize_level);
		Copy_Params(conv4_6_bias, Unicorn, quantize_level);
		Copy_Params(relu4_6_weights, Unicorn, quantize_level);
		Copy_Params(conv4_7_weights, Unicorn, quantize_level);
		Copy_Params(conv4_7_bias, Unicorn, quantize_level);
		Copy_Params(relu4_7_weights, Unicorn, quantize_level);
		Copy_Params(conv4_8_weights, Unicorn, quantize_level);
		Copy_Params(conv4_8_bias, Unicorn, quantize_level);
		Copy_Params(relu4_8_weights, Unicorn, quantize_level);
		Copy_Params(conv4_9_weights, Unicorn, quantize_level);
		Copy_Params(conv4_9_bias, Unicorn, quantize_level);
		Copy_Params(relu4_9_weights, Unicorn, quantize_level);
		Copy_Params(conv4_10_weights, Unicorn, quantize_level);
		Copy_Params(conv4_10_bias, Unicorn, quantize_level);
		Copy_Params(relu4_10_weights, Unicorn, quantize_level);
		Copy_Params(conv4_weights, Unicorn, quantize_level);
		Copy_Params(conv4_bias, Unicorn, quantize_level);
		Copy_Params(relu4_weights, Unicorn, quantize_level);
		Copy_Params(conv5_1_weights, Unicorn, quantize_level);
		Copy_Params(conv5_1_bias, Unicorn, quantize_level);
		Copy_Params(relu5_1_weights, Unicorn, quantize_level);
		Copy_Params(conv5_2_weights, Unicorn, quantize_level);
		Copy_Params(conv5_2_bias, Unicorn, quantize_level);
		Copy_Params(relu5_2_weights, Unicorn, quantize_level);
		Copy_Params(conv5_3_weights, Unicorn, quantize_level);
		Copy_Params(conv5_3_bias, Unicorn, quantize_level);
		Copy_Params(relu5_3_weights, Unicorn, quantize_level);
		Copy_Params(conv5_4_weights, Unicorn, quantize_level);
		Copy_Params(conv5_4_bias, Unicorn, quantize_level);
		Copy_Params(relu5_4_weights, Unicorn, quantize_level);
		Copy_Params(conv5_5_weights, Unicorn, quantize_level);
		Copy_Params(conv5_5_bias, Unicorn, quantize_level);
		Copy_Params(relu5_5_weights, Unicorn, quantize_level);
		Copy_Params(conv5_6_weights, Unicorn, quantize_level);
		Copy_Params(conv5_6_bias, Unicorn, quantize_level);
		Copy_Params(relu5_6_weights, Unicorn, quantize_level);
		Copy_Params(conv5_weights, Unicorn, quantize_level);
		Copy_Params(conv5_bias, Unicorn, quantize_level);
		Copy_Params(relu5_weights, Unicorn, quantize_level);
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
		Init_Flip_Params(fliper, false, true);
		Init_Concat_Params(concator, 0);
		Init_Conv_Params(conv1a, 3, 32, 3, 1, 0, true);
		Init_PReLU_Params(relu1a, 32, false);
		Init_Conv_Params(conv1b, 32, 64, 3, 1, 0, true);
		Init_PReLU_Params(relu1b, 64, false);
		Init_Pooling_Params(pool1b, 2, 2, 0, 0);
		Init_Conv_Params(conv2_1, 64, 64, 3, 1, 1, true);
		Init_PReLU_Params(relu2_1, 64, false);
		Init_Conv_Params(conv2_2, 64, 64, 3, 1, 1, true);
		Init_PReLU_Params(relu2_2, 64, false);
		Init_Eltwise_Params(res2_2, 0);
		Init_Conv_Params(conv2, 64, 128, 3, 1, 0, true);
		Init_PReLU_Params(relu2, 128, false);
		Init_Pooling_Params(pool2, 2, 2, 0, 0);
		Init_Conv_Params(conv3_1, 128, 128, 3, 1, 1, true);
		Init_PReLU_Params(relu3_1, 128, false);
		Init_Conv_Params(conv3_2, 128, 128, 3, 1, 1, true);
		Init_PReLU_Params(relu3_2, 128, false);
		Init_Eltwise_Params(res3_2, 0);
		Init_Conv_Params(conv3_3, 128, 128, 3, 1, 1, true);
		Init_PReLU_Params(relu3_3, 128, false);
		Init_Conv_Params(conv3_4, 128, 128, 3, 1, 1, true);
		Init_PReLU_Params(relu3_4, 128, false);
		Init_Eltwise_Params(res3_4, 0);
		Init_Conv_Params(conv3, 128, 256, 3, 1, 0, true);
		Init_PReLU_Params(relu3, 256, false);
		Init_Pooling_Params(pool3, 2, 2, 0, 0);
		Init_Conv_Params(conv4_1, 256, 256, 3, 1, 1, true);
		Init_PReLU_Params(relu4_1, 256, false);
		Init_Conv_Params(conv4_2, 256, 256, 3, 1, 1, true);
		Init_PReLU_Params(relu4_2, 256, false);
		Init_Eltwise_Params(res4_2, 0);
		Init_Conv_Params(conv4_3, 256, 256, 3, 1, 1, true);
		Init_PReLU_Params(relu4_3, 256, false);
		Init_Conv_Params(conv4_4, 256, 256, 3, 1, 1, true);
		Init_PReLU_Params(relu4_4, 256, false);
		Init_Eltwise_Params(res4_4, 0);
		Init_Conv_Params(conv4_5, 256, 256, 3, 1, 1, true);
		Init_PReLU_Params(relu4_5, 256, false);
		Init_Conv_Params(conv4_6, 256, 256, 3, 1, 1, true);
		Init_PReLU_Params(relu4_6, 256, false);
		Init_Eltwise_Params(res4_6, 0);
		Init_Conv_Params(conv4_7, 256, 256, 3, 1, 1, true);
		Init_PReLU_Params(relu4_7, 256, false);
		Init_Conv_Params(conv4_8, 256, 256, 3, 1, 1, true);
		Init_PReLU_Params(relu4_8, 256, false);
		Init_Eltwise_Params(res4_8, 0);
		Init_Conv_Params(conv4_9, 256, 256, 3, 1, 1, true);
		Init_PReLU_Params(relu4_9, 256, false);
		Init_Conv_Params(conv4_10, 256, 256, 3, 1, 1, true);
		Init_PReLU_Params(relu4_10, 256, false);
		Init_Eltwise_Params(res4_10, 0);
		Init_Conv_Params(conv4, 256, 512, 3, 1, 0, true);
		Init_PReLU_Params(relu4, 512, false);
		Init_Pooling_Params(pool4, 2, 2, 0, 0);
		Init_Conv_Params(conv5_1, 512, 512, 3, 1, 1, true);
		Init_PReLU_Params(relu5_1, 512, false);
		Init_Conv_Params(conv5_2, 512, 512, 3, 1, 1, true);
		Init_PReLU_Params(relu5_2, 512, false);
		Init_Eltwise_Params(res5_2, 0);
		Init_Conv_Params(conv5_3, 512, 512, 3, 1, 1, true);
		Init_PReLU_Params(relu5_3, 512, false);
		Init_Conv_Params(conv5_4, 512, 512, 3, 1, 1, true);
		Init_PReLU_Params(relu5_4, 512, false);
		Init_Eltwise_Params(res5_4, 0);
		Init_Conv_Params(conv5_5, 512, 512, 3, 1, 1, true);
		Init_PReLU_Params(relu5_5, 512, false);
		Init_Conv_Params(conv5_6, 512, 512, 3, 1, 1, true);
		Init_PReLU_Params(relu5_6, 512, false);
		Init_Eltwise_Params(res5_6, 0);
		Init_Conv_Params(conv5, 512, 512, 3, 1, 0, true);
		Init_PReLU_Params(relu5, 512, false);
		Init_Pooling_Params(pool5, 4, 4, 0, 1);
		Init_MirrorMax_Param(mirrmax, 0);
		Init_Normalize_Params(normalizer, 1, false);
	}


	unicorn_net::~unicorn_net()
	{
		delete fliper;
		delete concator;
		delete conv1a;
		delete relu1a;
		delete conv1b;
		delete relu1b;
		delete pool1b;
		delete conv2_1;
		delete relu2_1;
		delete conv2_2;
		delete relu2_2;
		delete res2_2;
		delete conv2;
		delete relu2;
		delete pool2;
		delete conv3_1;
		delete relu3_1;
		delete conv3_2;
		delete relu3_2;
		delete res3_2;
		delete conv3_3;
		delete relu3_3;
		delete conv3_4;
		delete relu3_4;
		delete res3_4;
		delete conv3;
		delete relu3;
		delete pool3;
		delete conv4_1;
		delete relu4_1;
		delete conv4_2;
		delete relu4_2;
		delete res4_2;
		delete conv4_3;
		delete relu4_3;
		delete conv4_4;
		delete relu4_4;
		delete res4_4;
		delete conv4_5;
		delete relu4_5;
		delete conv4_6;
		delete relu4_6;
		delete res4_6;
		delete conv4_7;
		delete relu4_7;
		delete conv4_8;
		delete relu4_8;
		delete res4_8;
		delete conv4_9;
		delete relu4_9;
		delete conv4_10;
		delete relu4_10;
		delete res4_10;
		delete conv4;
		delete relu4;
		delete pool4;
		delete conv5_1;
		delete relu5_1;
		delete conv5_2;
		delete relu5_2;
		delete res5_2;
		delete conv5_3;
		delete relu5_3;
		delete conv5_4;
		delete relu5_4;
		delete res5_4;
		delete conv5_5;
		delete relu5_5;
		delete conv5_6;
		delete relu5_6;
		delete res5_6;
		delete conv5;
		delete relu5;
		delete pool5;
		delete mirrmax;
		delete normalizer;

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

	void unicorn_net::Forward_cpu(const std::shared_ptr<tensor<float>> input_data)
	{
		//fliper->Forward_cpu(tensor_data, flip_top_data);
		//concator->Forward_cpu(std::vector<std::shared_ptr<tensor>>{tensor_data, flip_top_data}, concat_top_data);
		conv1a->Forward_cpu(input_data, conv1a_top_data);
		relu1a->Forward_cpu(conv1a_top_data);
		conv1b->Forward_cpu(conv1a_top_data, conv1b_top_data);
		relu1b->Forward_cpu(conv1b_top_data);
		pool1b->Forward_cpu(conv1b_top_data, pool1b_top_data);
		conv2_1->Forward_cpu(pool1b_top_data, conv2_1_top_data);
		relu2_1->Forward_cpu(conv2_1_top_data);
		conv2_2->Forward_cpu(conv2_1_top_data, conv2_2_top_data);
		relu2_2->Forward_cpu(conv2_2_top_data);
		res2_2->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{pool1b_top_data, conv2_2_top_data}, res2_2_top_data);
		conv2->Forward_cpu(res2_2_top_data, conv2_top_data);
		relu2->Forward_cpu(conv2_top_data);
		pool2->Forward_cpu(conv2_top_data, pool2_top_data);
		conv3_1->Forward_cpu(pool2_top_data, conv3_1_top_data);
		relu3_1->Forward_cpu(conv3_1_top_data);
		conv3_2->Forward_cpu(conv3_1_top_data, conv3_2_top_data);
		relu3_2->Forward_cpu(conv3_2_top_data);
		res3_2->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{pool2_top_data, conv3_2_top_data}, res3_2_top_data);
		conv3_3->Forward_cpu(res3_2_top_data, conv3_3_top_data);
		relu3_3->Forward_cpu(conv3_3_top_data);
		conv3_4->Forward_cpu(conv3_3_top_data, conv3_4_top_data);
		relu3_4->Forward_cpu(conv3_4_top_data);
		res3_4->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{res3_2_top_data, conv3_4_top_data}, res3_4_top_data);
		conv3->Forward_cpu(res3_4_top_data, conv3_top_data);
		relu3->Forward_cpu(conv3_top_data);
		pool3->Forward_cpu(conv3_top_data, pool3_top_data);
		conv4_1->Forward_cpu(pool3_top_data, conv4_1_top_data);
		relu4_1->Forward_cpu(conv4_1_top_data);
		conv4_2->Forward_cpu(conv4_1_top_data, conv4_2_top_data);
		relu4_2->Forward_cpu(conv4_2_top_data);
		res4_2->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{pool3_top_data, conv4_2_top_data}, res4_2_top_data);
		conv4_3->Forward_cpu(res4_2_top_data, conv4_3_top_data);
		relu4_3->Forward_cpu(conv4_3_top_data);
		conv4_4->Forward_cpu(conv4_3_top_data, conv4_4_top_data);
		relu4_4->Forward_cpu(conv4_4_top_data);
		res4_4->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{res4_2_top_data, conv4_4_top_data}, res4_4_top_data);
		conv4_5->Forward_cpu(res4_4_top_data, conv4_5_top_data);
		relu4_5->Forward_cpu(conv4_5_top_data);
		conv4_6->Forward_cpu(conv4_5_top_data, conv4_6_top_data);
		relu4_6->Forward_cpu(conv4_6_top_data);
		res4_6->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{res4_4_top_data, conv4_6_top_data}, res4_6_top_data);
		conv4_7->Forward_cpu(res4_6_top_data, conv4_7_top_data);
		relu4_7->Forward_cpu(conv4_7_top_data);
		conv4_8->Forward_cpu(conv4_7_top_data, conv4_8_top_data);
		relu4_8->Forward_cpu(conv4_8_top_data);
		res4_8->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{res4_6_top_data, conv4_8_top_data}, res4_8_top_data);
		conv4_9->Forward_cpu(res4_8_top_data, conv4_9_top_data);
		relu4_9->Forward_cpu(conv4_9_top_data);
		conv4_10->Forward_cpu(conv4_9_top_data, conv4_10_top_data);
		relu4_10->Forward_cpu(conv4_10_top_data);
		res4_10->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{res4_8_top_data, conv4_10_top_data}, res4_10_top_data);
		conv4->Forward_cpu(res4_10_top_data, conv4_top_data);
		relu4->Forward_cpu(conv4_top_data);
		pool4->Forward_cpu(conv4_top_data, pool4_top_data);
		conv5_1->Forward_cpu(pool4_top_data, conv5_1_top_data);
		relu5_1->Forward_cpu(conv5_1_top_data);
		conv5_2->Forward_cpu(conv5_1_top_data, conv5_2_top_data);
		relu5_2->Forward_cpu(conv5_2_top_data);
		res5_2->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{pool4_top_data, conv5_2_top_data}, res5_2_top_data);
		conv5_3->Forward_cpu(res5_2_top_data, conv5_3_top_data);
		relu5_3->Forward_cpu(conv5_3_top_data);
		conv5_4->Forward_cpu(conv5_3_top_data, conv5_4_top_data);
		relu5_4->Forward_cpu(conv5_4_top_data);
		res5_4->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{res5_2_top_data, conv5_4_top_data}, res5_4_top_data);
		conv5_5->Forward_cpu(res5_4_top_data, conv5_5_top_data);
		relu5_5->Forward_cpu(conv5_5_top_data);
		conv5_6->Forward_cpu(conv5_5_top_data, conv5_6_top_data);
		relu5_6->Forward_cpu(conv5_6_top_data);
		res5_6->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{res5_4_top_data, conv5_6_top_data}, res5_6_top_data);
		conv5->Forward_cpu(res5_6_top_data, conv5_top_data);
		relu5->Forward_cpu(conv5_top_data);
		pool5->Forward_cpu(conv5_top_data, pool5_top_data);
		//mirrmax->Forward_cpu(pool5_top_data, feature_top_data);
		calc_quality_score();
		normalizer->Forward_cpu(pool5_top_data);//feature_top_data
	}
	
#ifdef USE_CUDA
	void unicorn_net::Forward_native_gpu(const std::shared_ptr<tensor<float>> input_data)
	{
		/*fliper->Forward_native_gpu(tensor_data, flip_top_data);
		concator->Forward_native_gpu(std::vector<std::shared_ptr<tensor>>{tensor_data, flip_top_data}, concat_top_data);*/
		conv1a->Forward_native_gpu(cublas_handle_, input_data, conv1a_top_data);//concat_top_data
		relu1a->Forward_native_gpu(conv1a_top_data);
		conv1b->Forward_native_gpu(cublas_handle_, conv1a_top_data, conv1b_top_data);
		relu1b->Forward_native_gpu(conv1b_top_data);
		pool1b->Forward_native_gpu(conv1b_top_data, pool1b_top_data);
		conv2_1->Forward_native_gpu(cublas_handle_, pool1b_top_data, conv2_1_top_data);
		relu2_1->Forward_native_gpu(conv2_1_top_data);
		conv2_2->Forward_native_gpu(cublas_handle_, conv2_1_top_data, conv2_2_top_data);
		relu2_2->Forward_native_gpu(conv2_2_top_data);
		res2_2->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{pool1b_top_data, conv2_2_top_data}, res2_2_top_data);
		conv2->Forward_native_gpu(cublas_handle_, res2_2_top_data, conv2_top_data);
		relu2->Forward_native_gpu(conv2_top_data);
		pool2->Forward_native_gpu(conv2_top_data, pool2_top_data);
		conv3_1->Forward_native_gpu(cublas_handle_, pool2_top_data, conv3_1_top_data);
		relu3_1->Forward_native_gpu(conv3_1_top_data);
		conv3_2->Forward_native_gpu(cublas_handle_, conv3_1_top_data, conv3_2_top_data);
		relu3_2->Forward_native_gpu(conv3_2_top_data);
		res3_2->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{pool2_top_data, conv3_2_top_data}, res3_2_top_data);
		conv3_3->Forward_native_gpu(cublas_handle_, res3_2_top_data, conv3_3_top_data);
		relu3_3->Forward_native_gpu(conv3_3_top_data);
		conv3_4->Forward_native_gpu(cublas_handle_, conv3_3_top_data, conv3_4_top_data);
		relu3_4->Forward_native_gpu(conv3_4_top_data);
		res3_4->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res3_2_top_data, conv3_4_top_data}, res3_4_top_data);
		conv3->Forward_native_gpu(cublas_handle_, res3_4_top_data, conv3_top_data);
		relu3->Forward_native_gpu(conv3_top_data);
		pool3->Forward_native_gpu(conv3_top_data, pool3_top_data);
		conv4_1->Forward_native_gpu(cublas_handle_, pool3_top_data, conv4_1_top_data);
		relu4_1->Forward_native_gpu(conv4_1_top_data);
		conv4_2->Forward_native_gpu(cublas_handle_, conv4_1_top_data, conv4_2_top_data);
		relu4_2->Forward_native_gpu(conv4_2_top_data);
		res4_2->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{pool3_top_data, conv4_2_top_data}, res4_2_top_data);
		conv4_3->Forward_native_gpu(cublas_handle_, res4_2_top_data, conv4_3_top_data);
		relu4_3->Forward_native_gpu(conv4_3_top_data);
		conv4_4->Forward_native_gpu(cublas_handle_, conv4_3_top_data, conv4_4_top_data);
		relu4_4->Forward_native_gpu(conv4_4_top_data);
		res4_4->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res4_2_top_data, conv4_4_top_data}, res4_4_top_data);
		conv4_5->Forward_native_gpu(cublas_handle_, res4_4_top_data, conv4_5_top_data);
		relu4_5->Forward_native_gpu(conv4_5_top_data);
		conv4_6->Forward_native_gpu(cublas_handle_, conv4_5_top_data, conv4_6_top_data);
		relu4_6->Forward_native_gpu(conv4_6_top_data);
		res4_6->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res4_4_top_data, conv4_6_top_data}, res4_6_top_data);
		conv4_7->Forward_native_gpu(cublas_handle_, res4_6_top_data, conv4_7_top_data);
		relu4_7->Forward_native_gpu(conv4_7_top_data);
		conv4_8->Forward_native_gpu(cublas_handle_, conv4_7_top_data, conv4_8_top_data);
		relu4_8->Forward_native_gpu(conv4_8_top_data);
		res4_8->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res4_6_top_data, conv4_8_top_data}, res4_8_top_data);
		conv4_9->Forward_native_gpu(cublas_handle_, res4_8_top_data, conv4_9_top_data);
		relu4_9->Forward_native_gpu(conv4_9_top_data);
		conv4_10->Forward_native_gpu(cublas_handle_, conv4_9_top_data, conv4_10_top_data);
		relu4_10->Forward_native_gpu(conv4_10_top_data);
		res4_10->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res4_8_top_data, conv4_10_top_data}, res4_10_top_data);
		conv4->Forward_native_gpu(cublas_handle_, res4_10_top_data, conv4_top_data);
		relu4->Forward_native_gpu(conv4_top_data);
		pool4->Forward_native_gpu(conv4_top_data, pool4_top_data);
		conv5_1->Forward_native_gpu(cublas_handle_, pool4_top_data, conv5_1_top_data);
		relu5_1->Forward_native_gpu(conv5_1_top_data);
		conv5_2->Forward_native_gpu(cublas_handle_, conv5_1_top_data, conv5_2_top_data);
		relu5_2->Forward_native_gpu(conv5_2_top_data);
		res5_2->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{pool4_top_data, conv5_2_top_data}, res5_2_top_data);
		conv5_3->Forward_native_gpu(cublas_handle_, res5_2_top_data, conv5_3_top_data);
		relu5_3->Forward_native_gpu(conv5_3_top_data);
		conv5_4->Forward_native_gpu(cublas_handle_, conv5_3_top_data, conv5_4_top_data);
		relu5_4->Forward_native_gpu(conv5_4_top_data);
		res5_4->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res5_2_top_data, conv5_4_top_data}, res5_4_top_data);
		conv5_5->Forward_native_gpu(cublas_handle_, res5_4_top_data, conv5_5_top_data);
		relu5_5->Forward_native_gpu(conv5_5_top_data);
		conv5_6->Forward_native_gpu(cublas_handle_, conv5_5_top_data, conv5_6_top_data);
		relu5_6->Forward_native_gpu(conv5_6_top_data);
		res5_6->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res5_4_top_data, conv5_6_top_data}, res5_6_top_data);
		conv5->Forward_native_gpu(cublas_handle_, res5_6_top_data, conv5_top_data);
		relu5->Forward_native_gpu(conv5_top_data);
		pool5->Forward_native_gpu(conv5_top_data, pool5_top_data);
		//mirrmax->Forward_native_gpu(pool5_top_data, feature_top_data);
		calc_quality_score();
		normalizer->Forward_native_gpu(pool5_top_data);//feature_top_data
	}
#ifdef USE_CUDNN
	void unicorn_net::Forward_cudnn_gpu(const std::shared_ptr<tensor<float>> input_data)
	{
		conv1a->Forward_cudnn_gpu(input_data, conv1a_top_data);//concat_top_data
		relu1a->Forward_native_gpu(conv1a_top_data);
		conv1b->Forward_cudnn_gpu(conv1a_top_data, conv1b_top_data);
		relu1b->Forward_native_gpu(conv1b_top_data);
		pool1b->Forward_cudnn_gpu(conv1b_top_data, pool1b_top_data);
		conv2_1->Forward_cudnn_gpu(pool1b_top_data, conv2_1_top_data);
		relu2_1->Forward_native_gpu(conv2_1_top_data);
		conv2_2->Forward_cudnn_gpu(conv2_1_top_data, conv2_2_top_data);
		relu2_2->Forward_native_gpu(conv2_2_top_data);
		res2_2->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{pool1b_top_data, conv2_2_top_data}, res2_2_top_data);
		conv2->Forward_cudnn_gpu(res2_2_top_data, conv2_top_data);
		relu2->Forward_native_gpu(conv2_top_data);
		pool2->Forward_cudnn_gpu(conv2_top_data, pool2_top_data);
		conv3_1->Forward_cudnn_gpu(pool2_top_data, conv3_1_top_data);
		relu3_1->Forward_native_gpu(conv3_1_top_data);
		conv3_2->Forward_cudnn_gpu(conv3_1_top_data, conv3_2_top_data);
		relu3_2->Forward_native_gpu(conv3_2_top_data);
		res3_2->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{pool2_top_data, conv3_2_top_data}, res3_2_top_data);
		conv3_3->Forward_cudnn_gpu(res3_2_top_data, conv3_3_top_data);
		relu3_3->Forward_native_gpu(conv3_3_top_data);
		conv3_4->Forward_cudnn_gpu(conv3_3_top_data, conv3_4_top_data);
		relu3_4->Forward_native_gpu(conv3_4_top_data);
		res3_4->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res3_2_top_data, conv3_4_top_data}, res3_4_top_data);
		conv3->Forward_cudnn_gpu(res3_4_top_data, conv3_top_data);
		relu3->Forward_native_gpu(conv3_top_data);
		pool3->Forward_cudnn_gpu(conv3_top_data, pool3_top_data);
		conv4_1->Forward_cudnn_gpu(pool3_top_data, conv4_1_top_data);
		relu4_1->Forward_native_gpu(conv4_1_top_data);
		conv4_2->Forward_cudnn_gpu(conv4_1_top_data, conv4_2_top_data);
		relu4_2->Forward_native_gpu(conv4_2_top_data);
		res4_2->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{pool3_top_data, conv4_2_top_data}, res4_2_top_data);
		conv4_3->Forward_cudnn_gpu(res4_2_top_data, conv4_3_top_data);
		relu4_3->Forward_native_gpu(conv4_3_top_data);
		conv4_4->Forward_cudnn_gpu(conv4_3_top_data, conv4_4_top_data);
		relu4_4->Forward_native_gpu(conv4_4_top_data);
		res4_4->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res4_2_top_data, conv4_4_top_data}, res4_4_top_data);
		conv4_5->Forward_cudnn_gpu(res4_4_top_data, conv4_5_top_data);
		relu4_5->Forward_native_gpu(conv4_5_top_data);
		conv4_6->Forward_cudnn_gpu(conv4_5_top_data, conv4_6_top_data);
		relu4_6->Forward_native_gpu(conv4_6_top_data);
		res4_6->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res4_4_top_data, conv4_6_top_data}, res4_6_top_data);
		conv4_7->Forward_cudnn_gpu(res4_6_top_data, conv4_7_top_data);
		relu4_7->Forward_native_gpu(conv4_7_top_data);
		conv4_8->Forward_cudnn_gpu(conv4_7_top_data, conv4_8_top_data);
		relu4_8->Forward_native_gpu(conv4_8_top_data);
		res4_8->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res4_6_top_data, conv4_8_top_data}, res4_8_top_data);
		conv4_9->Forward_cudnn_gpu(res4_8_top_data, conv4_9_top_data);
		relu4_9->Forward_native_gpu(conv4_9_top_data);
		conv4_10->Forward_cudnn_gpu(conv4_9_top_data, conv4_10_top_data);
		relu4_10->Forward_native_gpu(conv4_10_top_data);
		res4_10->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res4_8_top_data, conv4_10_top_data}, res4_10_top_data);
		conv4->Forward_cudnn_gpu(res4_10_top_data, conv4_top_data);
		relu4->Forward_native_gpu(conv4_top_data);
		pool4->Forward_cudnn_gpu(conv4_top_data, pool4_top_data);
		conv5_1->Forward_cudnn_gpu(pool4_top_data, conv5_1_top_data);
		relu5_1->Forward_native_gpu(conv5_1_top_data);
		conv5_2->Forward_cudnn_gpu(conv5_1_top_data, conv5_2_top_data);
		relu5_2->Forward_native_gpu(conv5_2_top_data);
		res5_2->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{pool4_top_data, conv5_2_top_data}, res5_2_top_data);
		conv5_3->Forward_cudnn_gpu(res5_2_top_data, conv5_3_top_data);
		relu5_3->Forward_native_gpu(conv5_3_top_data);
		conv5_4->Forward_cudnn_gpu(conv5_3_top_data, conv5_4_top_data);
		relu5_4->Forward_native_gpu(conv5_4_top_data);
		res5_4->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res5_2_top_data, conv5_4_top_data}, res5_4_top_data);
		conv5_5->Forward_cudnn_gpu(res5_4_top_data, conv5_5_top_data);
		relu5_5->Forward_native_gpu(conv5_5_top_data);
		conv5_6->Forward_cudnn_gpu(conv5_5_top_data, conv5_6_top_data);
		relu5_6->Forward_native_gpu(conv5_6_top_data);
		res5_6->Forward_native_gpu(cublas_handle_, std::vector<std::shared_ptr<tensor<float>>>{res5_4_top_data, conv5_6_top_data}, res5_6_top_data);
		conv5->Forward_cudnn_gpu(res5_6_top_data, conv5_top_data);
		relu5->Forward_native_gpu(conv5_top_data);
		pool5->Forward_cudnn_gpu(conv5_top_data, pool5_top_data);
		normalizer->Forward_native_gpu(pool5_top_data);//feature_top_data
	}
#endif
#endif

	void unicorn_net::Forward(const std::shared_ptr<tensor<float>> input_data)
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


