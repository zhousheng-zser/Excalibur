#ifndef _DAILIR_ONET_NIR_HPP_
#define _DAILIR_ONET_NIR_HPP_
#include "../Excalibur/support_layers.hpp"
#include "onet_mobile_nir_data.hpp"

using namespace glasssix::excalibur;

namespace glasssix
{
	namespace longinus
	{
		class onet_mobile_nir
		{
			Declear_Params(conv1);
			Declear_Params(relu1);
			Declear_Params(conv1_dw);
			Declear_Params(relu1_dw);
			Declear_Params(conv2_ex);
			Declear_Params(relu2_ex);
			Declear_Params(conv2_dw);
			Declear_Params(relu2_dw);
			Declear_Params(conv2_em);
			Declear_Params(conv2_1_ex);
			Declear_Params(relu2_1_ex);
			Declear_Params(conv2_1_dw);
			Declear_Params(relu2_1_dw);
			Declear_Params(conv2_1_em);
			Declear_Params(conv3_ex);
			Declear_Params(relu3_ex);
			Declear_Params(conv3_dw);
			Declear_Params(relu3_dw);
			Declear_Params(conv3_em);
			Declear_Params(conv3_1_ex);
			Declear_Params(relu3_1_ex);
			Declear_Params(conv3_1_dw);
			Declear_Params(relu3_1_dw);
			Declear_Params(conv3_1_em);
			Declear_Params(conv4_ex);
			Declear_Params(relu4_ex);
			Declear_Params(conv4_dw);
			Declear_Params(relu4_dw);
			Declear_Params(conv4_em);
			Declear_Params(conv4_1_ex);
			Declear_Params(relu4_1_ex);
			Declear_Params(conv4_1_dw);
			Declear_Params(relu4_1_dw);
			Declear_Params(conv4_1_em);
			Declear_Params(conv5_ex);
			Declear_Params(relu5_ex);
			Declear_Params(conv5_dw);
			Declear_Params(conv6_1);
			Declear_Params(conv6_2);
			Declear_Params(conv6_3);
			Declear_Params(conv6_4);

			Declear_Opration(baseconv, conv1);
			Neuron_Name(conv1);
			Declear_Opration(prelu, relu1);
			Neuron_Name(relu1);
			Declear_Opration(baseconv, conv1_dw);
			Neuron_Name(conv1_dw);
			Declear_Opration(prelu, relu1_dw);
			Neuron_Name(relu1_dw);
			Declear_Opration(baseconv, conv2_ex);
			Neuron_Name(conv2_ex);
			Declear_Opration(prelu, relu2_ex);
			Neuron_Name(relu2_ex);
			Declear_Opration(baseconv, conv2_dw);
			Neuron_Name(conv2_dw);
			Declear_Opration(prelu, relu2_dw);
			Neuron_Name(relu2_dw);
			Declear_Opration(baseconv, conv2_em);
			Neuron_Name(conv2_em);
			Declear_Opration(baseconv, conv2_1_ex);
			Neuron_Name(conv2_1_ex);
			Declear_Opration(prelu, relu2_1_ex);
			Neuron_Name(relu2_1_ex);
			Declear_Opration(baseconv, conv2_1_dw);
			Neuron_Name(conv2_1_dw);
			Declear_Opration(prelu, relu2_1_dw);
			Neuron_Name(relu2_1_dw);
			Declear_Opration(baseconv, conv2_1_em);
			Neuron_Name(conv2_1_em);
			Declear_Opration(eltwise, res2);
			Neuron_Name(res2);
			Declear_Opration(baseconv, conv3_ex);
			Neuron_Name(conv3_ex);
			Declear_Opration(prelu, relu3_ex);
			Neuron_Name(relu3_ex);
			Declear_Opration(baseconv, conv3_dw);
			Neuron_Name(conv3_dw);
			Declear_Opration(prelu, relu3_dw);
			Neuron_Name(relu3_dw);
			Declear_Opration(baseconv, conv3_em);
			Neuron_Name(conv3_em);
			Declear_Opration(baseconv, conv3_1_ex);
			Neuron_Name(conv3_1_ex);
			Declear_Opration(prelu, relu3_1_ex);
			Neuron_Name(relu3_1_ex);
			Declear_Opration(baseconv, conv3_1_dw);
			Neuron_Name(conv3_1_dw);
			Declear_Opration(prelu, relu3_1_dw);
			Neuron_Name(relu3_1_dw);
			Declear_Opration(baseconv, conv3_1_em);
			Neuron_Name(conv3_1_em);
			Declear_Opration(eltwise, res3);
			Neuron_Name(res3);
			Declear_Opration(baseconv, conv4_ex);
			Neuron_Name(conv4_ex);
			Declear_Opration(prelu, relu4_ex);
			Neuron_Name(relu4_ex);
			Declear_Opration(baseconv, conv4_dw);
			Neuron_Name(conv4_dw);
			Declear_Opration(prelu, relu4_dw);
			Neuron_Name(relu4_dw);
			Declear_Opration(baseconv, conv4_em);
			Neuron_Name(conv4_em);
			Declear_Opration(baseconv, conv4_1_ex);
			Neuron_Name(conv4_1_ex);
			Declear_Opration(prelu, relu4_1_ex);
			Neuron_Name(relu4_1_ex);
			Declear_Opration(baseconv, conv4_1_dw);
			Neuron_Name(conv4_1_dw);
			Declear_Opration(prelu, relu4_1_dw);
			Neuron_Name(relu4_1_dw);
			Declear_Opration(baseconv, conv4_1_em);
			Neuron_Name(conv4_1_em);
			Declear_Opration(eltwise, res4);
			Neuron_Name(res4);
			Declear_Opration(baseconv, conv5_ex);
			Neuron_Name(conv5_ex);
			Declear_Opration(prelu, relu5_ex);
			Neuron_Name(relu5_ex);
			Declear_Opration(baseconv, conv5_dw);
			Neuron_Name(conv5_dw);
			Declear_Opration(inner_product, conv6_1);
			Neuron_Name(conv6_1);
			Declear_Opration(inner_product, conv6_2);
			Neuron_Name(conv6_2);
			Declear_Opration(inner_product, conv6_3);
			Neuron_Name(conv6_3);
			Declear_Opration(inner_product, conv6_4);
			Neuron_Name(conv6_4);
			Declear_Opration(sigmoid, prob1);
			Neuron_Name(prob1);

			//
			std::shared_ptr<tensor<float>> tensor_data;
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
			onet_mobile_nir(int device);
			~onet_mobile_nir();
			void Forward(const std::shared_ptr<tensor<float>> input_data);
		};
	}
}
#endif