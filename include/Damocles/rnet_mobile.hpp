#ifndef _DAILIR_RNET_MOBILE_HPP_
#define _DAILIR_RNET_MOBILE_HPP_
#include "Excalibur/support_layers.hpp"
#include "Primitives/tensor.hpp"
#include "rnet_mobile_data.hpp"

using namespace glasssix::excalibur;
using glasssix::memory::tensor;

namespace glasssix
{
	namespace longinus
	{
		class rnet_mobile
		{
			Declear_Params(conv1);
			Declear_Params(prelu1);
			Declear_Params(conv2_sep);
			Declear_Params(prelu2);
			Declear_Params(conv3_sep);
			Declear_Params(prelu3);
			Declear_Params(conv4_dw);
			Declear_Params(prelu4_dw);
			Declear_Params(conv4_sep);
			Declear_Params(prelu4);
			Declear_Params(conv5_dw);
			Declear_Params(prelu5_dw);
			Declear_Params(conv5_1);
			Declear_Params(conv5_1_bn);
			Declear_Params(conv5_1_sc);
			Declear_Params(conv5_2);
			Declear_Params(conv5_2_bn);
			Declear_Params(conv5_2_sc);

			//
			std::shared_ptr<tensor<float>> tensor_data;
			Declear_Opration(baseconv, conv1);
			Neuron_Name(conv1);
			Declear_Opration(prelu, prelu1);
			Neuron_Name(prelu1);
			Declear_Opration(pooling, pool1);
			Neuron_Name(pool1);
			Declear_Opration(baseconv, conv2_sep);
			Neuron_Name(conv2_sep);
			Declear_Opration(prelu, prelu2);
			Neuron_Name(prelu2);
			Declear_Opration(pooling, pool2);
			Neuron_Name(pool2);
			Declear_Opration(baseconv, conv3_sep);
			Neuron_Name(conv3_sep);
			Declear_Opration(prelu, prelu3);
			Neuron_Name(prelu3);
			Declear_Opration(baseconv, conv4_dw);
			Neuron_Name(conv4_dw);
			Declear_Opration(prelu, prelu4_dw);
			Neuron_Name(prelu4_dw);
			Declear_Opration(baseconv, conv4_sep);
			Neuron_Name(conv4_sep);
			Declear_Opration(prelu, prelu4);
			Neuron_Name(prelu4);
			Declear_Opration(baseconv, conv5_dw);
			Neuron_Name(conv5_dw);
			Declear_Opration(prelu, prelu5_dw);
			Neuron_Name(prelu5_dw);
			Declear_Opration(inner_product, conv5_1);
			Neuron_Name(conv5_1);
			Declear_Opration(batchnorm_arm, conv5_1_bn);
			Neuron_Name(conv5_1_bn);
			Declear_Opration(scale_arm, conv5_1_sc);
			Neuron_Name(conv5_1_sc);
			Declear_Opration(softmax, cls_prob);
			Neuron_Name(cls_prob);
			Declear_Opration(inner_product, conv5_2);
			Neuron_Name(conv5_2);
			Declear_Opration(batchnorm_arm, conv5_2_bn);
			Neuron_Name(conv5_2_bn);
			Declear_Opration(scale_arm, conv5_2_sc);
			Neuron_Name(conv5_2_sc);
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
			rnet_mobile(int device);
			~rnet_mobile();
			void Forward(const std::shared_ptr<tensor<float>> input_data);
		};
	}
}

#endif