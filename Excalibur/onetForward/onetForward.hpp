#ifndef _ONET_FORWARD_HPP_
#define _ONET_FORWARD_HPP_

#include "onetForwardData.hpp"
#include "../Excalibur/support_layers.hpp"
#include "../Excalibur/tensor_operation_cpu.hpp"
#include "../Excalibur/ftensor.hpp"
#include "../Excalibur/utensor.hpp"

using namespace excalibur;

namespace glasssix
{
	class onetForward
	{
        Declear_Params(conv1_weights);
        Declear_Params(conv1_bias);
        Declear_Params(prelu1_weights);
		Declear_Params(conv1_dw_weights);
		Declear_Params(conv1_dw_bias);
		Declear_Params(prelu1_dw_weights);
        Declear_Params(conv2_weights);
        Declear_Params(conv2_bias);
		Declear_Params(conv2_dw_weights);
		Declear_Params(conv2_dw_bias);
		Declear_Params(prelu2_dw_weights);
		Declear_Params(conv3_weights);
		Declear_Params(conv3_bias);
		Declear_Params(conv3_dw_weights);
		Declear_Params(conv3_dw_bias);
		Declear_Params(prelu3_dw_weights);
		Declear_Params(conv4_weights);
		Declear_Params(conv4_bias);
		Declear_Params(conv4_dw_weights);
		Declear_Params(conv4_dw_bias);
		Declear_Params(prelu4_dw_weights);
		Declear_Params(conv5_weights);
		Declear_Params(conv5_bias);
		Declear_Params(prelu5_weights);
		Declear_Params(conv6_1_weights);
		Declear_Params(conv6_1_bias);
		Declear_Params(conv6_2_weights);
		Declear_Params(conv6_2_bias);
		Declear_Params(conv6_3_weights);
		Declear_Params(conv6_3_bias);

		//
		int device_;
		std::shared_ptr<tensor<float>> tensor_data = nullptr;
		//

		Declear_Opration(convolution, conv1);
		Neuron_Name(conv1);
		Declear_Opration(prelu, prelu1);
		Neuron_Name(prelu1);
		Declear_Opration(convolution, conv1_dw);
		Neuron_Name(conv1_dw);
		Declear_Opration(prelu, prelu1_dw);
		Neuron_Name(prelu1_dw);
		Declear_Opration(convolution, conv2);
		Neuron_Name(conv2);
		Declear_Opration(convolution, conv2_dw);
		Neuron_Name(conv2_dw);
		Declear_Opration(prelu, prelu2_dw);
		Neuron_Name(prelu2_dw);
		Declear_Opration(convolution, conv3);
		Neuron_Name(conv3);
		Declear_Opration(convolution, conv3_dw);
		Neuron_Name(conv3_dw);
		Declear_Opration(prelu, prelu3_dw);
		Neuron_Name(prelu3_dw);
		Declear_Opration(convolution, conv4);
		Neuron_Name(conv4);
		Declear_Opration(convolution, conv4_dw);
		Neuron_Name(conv4_dw);
		Declear_Opration(prelu, prelu4_dw);
		Neuron_Name(prelu4_dw);
		Declear_Opration(inner_product, conv5);
		Neuron_Name(conv5);
		Declear_Opration(prelu, prelu5);
		Neuron_Name(prelu5);
		Declear_Opration(inner_product, conv6_1);
		Neuron_Name(conv6_1);
		Declear_Opration(sigmoid, sigmoid1);
		Neuron_Name(sigmoid1);
		Declear_Opration(inner_product, conv6_2);
		Neuron_Name(conv6_2);
		Declear_Opration(inner_product, conv6_3);
		Neuron_Name(conv6_3);

#ifdef USE_CUDA
		cublasHandle_t cublas_handle_ = nullptr;
		void Forward_native_gpu(const std::shared_ptr<tensor<float>> input_data);
#ifdef USE_CUDNN
		void Forward_cudnn_gpu(const std::shared_ptr<tensor<float>> input_data);
#endif 
#endif
		void Forward_cpu(const std::shared_ptr<tensor<float>> input_data);

	public:
		onetForward(int device);
		~onetForward();

		void Forward(const std::shared_ptr<tensor<float>> input_data)
		{
			tensor_operation_cpu::preprocess_tensors_cpu(input_data);

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

		void Forward(ftensor* input_data)
		{
			tensor_operation_cpu::preprocess_tensors_cpu(input_data->getdata());
			tensor_data = std::make_shared<tensor<float>>(*input_data->getdata());
			Forward(tensor_data);
		}

		void Forward(utensor* input_data)
		{
			ftensor * f_input_data = new ftensor(input_data->getdata()->data_shape(), input_data->getdata()->device());
			tensor_operation_cpu::type_convertor_cpu(input_data->getdata(), f_input_data->getdata());
			tensor_operation_cpu::preprocess_tensors_cpu(f_input_data->getdata());
			tensor_data = std::make_shared<tensor<float>>(*f_input_data->getdata());
			Forward(tensor_data);
			delete f_input_data;
		}

		static int get_input_channel()
		{
			return 1;
		}
		static int get_input_width()
		{
			return 48;
		}
		static int get_input_height()
		{
			return 48;
		}
	};
}

#endif // !_ONET_FORWARD_HPP_