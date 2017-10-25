#pragma once
#ifndef _MTCNN_PNET_HPP_
#define _MTCNN_PNET_HPP_
#include "../Excalibur/io.hpp"
#include "../Excalibur/support_layers.hpp"
#define Neuron_Name(name) private: \
std::shared_ptr<tensor> name##_top_data = nullptr;\
public: std::shared_ptr<tensor> get_##name(){\
return name##_top_data;\
}\
private:

#define  Declear_Opration(op, name) op *##name;

using namespace excalibur;

namespace fastface
{
	class mtcnn_pnet
	{
		std::vector<float*> conv1_para;
		std::vector<float*> prelu1_para;
		std::vector<float*> conv2_para;
		std::vector<float*> prelu2_para;
		std::vector<float*> conv3_para;
		std::vector<float*> prelu3_para;
		std::vector<float*> conv4_1_para;
		std::vector<float*> conv4_2_para;
		//
		convolution* conv1;
		prelu *prelu1;
		pooling *pool1;
		convolution *conv2;
		prelu *prelu2;
		convolution *conv3;
		prelu *prelu3;
		convolution *conv4_1;
		convolution *conv4_2;
		softmax *prob1;
		//
		std::shared_ptr<tensor> tensor_data = nullptr;
		Neuron_Name(conv1)
		Neuron_Name(pool1)
		Neuron_Name(conv2)
		Neuron_Name(conv3)
		Neuron_Name(conv4_1)
		Neuron_Name(conv4_2)
		Neuron_Name(prob1)
		//
		int device_;
		void Forward_cpu(const std::shared_ptr<tensor> input_data);
#ifdef USE_CUDA
		cublasHandle_t cublas_handle_ = nullptr;
		void Forward_native_gpu(const std::shared_ptr<tensor> input_data);
#endif
		
	public:
		mtcnn_pnet(int device);
		~mtcnn_pnet();
		void Forward(const std::shared_ptr<tensor> input_data);
	};
}

#endif