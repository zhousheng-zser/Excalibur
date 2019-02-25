
#ifndef _MTCNN_RNET_HPP_
#define _MTCNN_RNET_HPP_
#include "../Excalibur/io.hpp"
#include "../Excalibur/support_layers.hpp"
#include "mtcnn_rnet_data.hpp"

using namespace glasssix::excalibur;

namespace glasssix
{
	namespace longinus
	{
		class mtcnn_rnet
		{
			Declear_Params(conv1_weights);
			Declear_Params(conv1_bias);
			Declear_Params(prelu1_weights);
			Declear_Params(conv2_weights);
			Declear_Params(conv2_bias);
			Declear_Params(prelu2_weights);
			Declear_Params(conv3_weights);
			Declear_Params(conv3_bias);
			Declear_Params(prelu3_weights);
			Declear_Params(conv4_weights);
			Declear_Params(conv4_bias);
			Declear_Params(prelu4_weights);
			Declear_Params(conv5_1_weights);
			Declear_Params(conv5_1_bias);
			Declear_Params(conv5_2_weights);
			Declear_Params(conv5_2_bias);
			Declear_Params(conv5_3_weights);
			Declear_Params(conv5_3_bias);
			//
			std::shared_ptr<tensor<float>> tensor_data = nullptr;
			Declear_Opration(convolution, conv1);
			Neuron_Name(conv1);
			Declear_Opration(prelu, prelu1);
			Neuron_Name(prelu1);
			Declear_Opration(pooling, pool1);
			Neuron_Name(pool1);
			Declear_Opration(convolution, conv2);
			Neuron_Name(conv2);
			Declear_Opration(prelu, prelu2);
			Neuron_Name(prelu2);
			Declear_Opration(pooling, pool2);
			Neuron_Name(pool2);
			Declear_Opration(convolution, conv3);
			Neuron_Name(conv3);
			Declear_Opration(prelu, prelu3);
			Neuron_Name(prelu3);
			Declear_Opration(inner_product, conv4);
			Neuron_Name(conv4);
			Declear_Opration(prelu, prelu4);
			Neuron_Name(prelu4);
			Declear_Opration(inner_product, conv5_1);
			Neuron_Name(conv5_1);
			Declear_Opration(inner_product, conv5_2);
			Neuron_Name(conv5_2);
			Declear_Opration(softmax, prob1);
			Neuron_Name(prob1);
			//
			int device_;
			void Forward_cpu(const std::shared_ptr<tensor<float>> input_data);
#ifdef USE_CUDA
			cublasHandle_t cublas_handle_ = nullptr;
			void Forward_native_gpu(const std::shared_ptr<tensor<float>> input_data);
#ifdef USE_CUDNN
			//cudnnHandle_t cudnn_handle_ = nullptr;
			void Forward_cudnn_gpu(const std::shared_ptr<tensor<float>> input_data);
#endif 
#endif
		public:
			mtcnn_rnet(int device);
			~mtcnn_rnet();
			void Forward(const std::shared_ptr<tensor<float>> input_data);
		};
	}
}

#endif