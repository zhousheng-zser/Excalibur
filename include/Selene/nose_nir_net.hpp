#ifndef _NOSE_NIR_NET_HPP_
#define _NOSE_NIR_NET_HPP_

#include "selene.hpp"
#include "Excalibur/support_layers.hpp"
#include "Excalibur/tensor_operation_cpu.hpp"
#include "Excalibur/tensor_operation_gpu.hpp"

using namespace glasssix::excalibur;

namespace glasssix
{
	namespace longinus
	{
		class Nose_nir_net
		{
			Declear_Params(conv1);
			Declear_Params(prelu1);
			Declear_Params(conv1_dw);
			Declear_Params(prelu1_dw);
			Declear_Params(conv2);
			Declear_Params(conv2_dw);
			Declear_Params(prelu2_dw);
			Declear_Params(conv3);
			Declear_Params(conv3_dw);
			Declear_Params(prelu3_dw);
			Declear_Params(conv4);
			Declear_Params(conv4_dw);
			Declear_Params(prelu4_dw);
			Declear_Params(conv5);
			Declear_Params(prelu5);
			Declear_Params(conv6_1);

			Declear_Opration(baseconv, conv1);
			Neuron_Name(conv1);
			Declear_Opration(prelu, prelu1);
			Neuron_Name(prelu1);
			Declear_Opration(baseconv, conv1_dw);
			Neuron_Name(conv1_dw);
			Declear_Opration(prelu, prelu1_dw);
			Neuron_Name(prelu1_dw);
			Declear_Opration(baseconv, conv2);
			Neuron_Name(conv2);
			Declear_Opration(baseconv, conv2_dw);
			Neuron_Name(conv2_dw);
			Declear_Opration(prelu, prelu2_dw);
			Neuron_Name(prelu2_dw);
			Declear_Opration(baseconv, conv3);
			Neuron_Name(conv3);
			Declear_Opration(baseconv, conv3_dw);
			Neuron_Name(conv3_dw);
			Declear_Opration(prelu, prelu3_dw);
			Neuron_Name(prelu3_dw);
			Declear_Opration(baseconv, conv4);
			Neuron_Name(conv4);
			Declear_Opration(baseconv, conv4_dw);
			Neuron_Name(conv4_dw);
			Declear_Opration(prelu, prelu4_dw);
			Neuron_Name(prelu4_dw);
			Declear_Opration(inner_product, conv5);
			Neuron_Name(conv5);
			Declear_Opration(prelu, prelu5);
			Neuron_Name(prelu5);
			Declear_Opration(inner_product, conv6_1);
			Neuron_Name(conv6_1);
			Declear_Opration(sigmoid, cls_loss);
			Neuron_Name(cls_loss);


			//
			int device_;
			bool cudnn_ready_ = false;
			bool int8_quantization_ = false;
			std::shared_ptr<memory::tensor<float>> tensor_float_data = nullptr;
			std::shared_ptr<memory::tensor<unsigned char>> tensor_unsigned_char_data = nullptr;
			//

#ifdef USE_CUDA
			cublasHandle_t cublas_handle_ = nullptr;
			void Forward_gpu_native(const std::shared_ptr<memory::tensor<float>> input_data);
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle_ = nullptr;
			void Forward_gpu_cudnn(const std::shared_ptr<memory::tensor<float>> input_data);
#endif 
#endif
			void Forward_cpu(const std::shared_ptr<memory::tensor<float>> input_data);

		public:
			Nose_nir_net(int device);
			virtual ~Nose_nir_net();

			void Forward(const std::shared_ptr<memory::tensor<unsigned char>> input_data, int order = 0)
			{
				float means[3] = { 127.5f, 127.5f, 127.5f };
				float var = 0.0078125;
				if (device_ < 0)
				{
					tensor_operation_cpu::preprocess_tensors_cpu(input_data, tensor_float_data, means, var);
					Forward_cpu(tensor_float_data);
				}
				else
				{
#ifdef USE_CUDA
					tensor_operation_gpu::preprocess_tensors_gpu(input_data, tensor_float_data, means, var);
#ifdef USE_CUDNN
					Forward_gpu_cudnn(tensor_float_data);
					return;
#endif
					Forward_gpu_native(tensor_float_data);
					return;
#else
					NO_GPU;
#endif
				}
			}

		};
	}
}
#endif