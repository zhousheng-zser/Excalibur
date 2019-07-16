
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
			Declear_Params(conv1);
			Declear_Params(prelu1);
			Declear_Params(conv2);
			Declear_Params(prelu2);
			Declear_Params(conv3);
			Declear_Params(prelu3);
			Declear_Params(conv4);
			Declear_Params(prelu4);
			Declear_Params(conv5_1);
			Declear_Params(conv5_2);
			Declear_Params(conv5_3);
			//
			std::shared_ptr<tensor<float>> tensor_data;
			Declear_Opration(baseconv, conv1);
			Neuron_Name(conv1);
			Declear_Opration(prelu, prelu1);
			Neuron_Name(prelu1);
			Declear_Opration(pooling, pool1);
			Neuron_Name(pool1);
			Declear_Opration(baseconv, conv2);
			Neuron_Name(conv2);
			Declear_Opration(prelu, prelu2);
			Neuron_Name(prelu2);
			Declear_Opration(pooling, pool2);
			Neuron_Name(pool2);
			Declear_Opration(baseconv, conv3);
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
			bool cudnn_ready_ = false;
			bool int8_quantization_ = false;
			void Forward_cpu(const std::shared_ptr<tensor<float>> input_data);
#ifdef USE_CUDA
			cublasHandle_t cublas_handle_ = nullptr;
			void Forward_gpu_native(const std::shared_ptr<tensor<float>> input_data);
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle_ = nullptr;
			void Forward_gpu_cudnn(const std::shared_ptr<tensor<float>> input_data);
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