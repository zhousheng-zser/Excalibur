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
		/*std::shared_ptr<tensor> conv1_top_data = nullptr;
		std::shared_ptr<tensor> pool1_top_data = nullptr;
		std::shared_ptr<tensor> conv2_top_data = nullptr;
		std::shared_ptr<tensor> conv3_top_data = nullptr;*/
		//Neuron_Name(tensor_data);
		Neuron_Name(conv1);
		Neuron_Name(pool1);
		Neuron_Name(conv2);
		Neuron_Name(conv3);
		Neuron_Name(conv4_1);
		std::shared_ptr<tensor> conv4_2_top_data = nullptr;
		std::shared_ptr<tensor> prob1_top_data = nullptr;
		//
		int device_;
		cublasHandle_t cublas_handle_ = nullptr;
	public:
		mtcnn_pnet();
		~mtcnn_pnet();
		void Forward_cpu(const std::shared_ptr<tensor> input_data);
#ifdef USE_CUDA
		void Forward_native_gpu(const std::shared_ptr<tensor> input_data);
#endif
		std::shared_ptr<tensor> get_prob1();
		std::shared_ptr<tensor> get_conv4_2();
	};
}

#endif