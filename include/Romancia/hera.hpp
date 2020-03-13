#ifndef _HERA_HPP_
#define _HERA_HPP_

#include "../Excalibur/support_layers.hpp"
#include "../Excalibur/tensor_operation_cpu.hpp"
#include "../Excalibur/tensor_operation_gpu.hpp"
using namespace glasssix::excalibur;

namespace glasssix
{
	namespace longinus
	{
		class Hera
		{
			Declear_Params(conv1);
			Declear_Params(convolution1);
			Declear_Params(convolutiondepthwise1);
			Declear_Params(convolution2);
			Declear_Params(convolution3);
			Declear_Params(convolutiondepthwise2);
			Declear_Params(convolution4);
			Declear_Params(convolution5);
			Declear_Params(convolutiondepthwise3);
			Declear_Params(convolution6);
			Declear_Params(convolution7);
			Declear_Params(convolutiondepthwise4);
			Declear_Params(convolution8);
			Declear_Params(convolution9);
			Declear_Params(convolutiondepthwise5);
			Declear_Params(convolution10);
			Declear_Params(convolution11);
			Declear_Params(convolutiondepthwise6);
			Declear_Params(convolution12);
			Declear_Params(fc1);
			Declear_Params(fc2);
			Declear_Params(fc3);
			Declear_Params(fc4);

			//
			int device_;
			bool cudnn_ready_ = false;
			bool int8_quantization_ = false;
			std::shared_ptr<tensor<float>> tensor_float_data = nullptr;
			std::shared_ptr<tensor<unsigned char>> tensor_unsigned_char_data = nullptr;
			//

			Declear_Opration(baseconv, conv1);
			Neuron_Name(conv1);
			Declear_Opration(prelu, relu1);
			Neuron_Name(relu1);
			Declear_Opration(baseconv, convolution1);
			Neuron_Name(convolution1);
			Declear_Opration(prelu, relu1_1);
			Neuron_Name(relu1_1);
			Declear_Opration(baseconv, convolutiondepthwise1);
			Neuron_Name(convolutiondepthwise1);
			Declear_Opration(prelu, relu2);
			Neuron_Name(relu2);
			Declear_Opration(baseconv, convolution2);
			Neuron_Name(convolution2);
			Declear_Opration(baseconv, convolution3);
			Neuron_Name(convolution3);
			Declear_Opration(prelu, relu3);
			Neuron_Name(relu3);
			Declear_Opration(baseconv, convolutiondepthwise2);
			Neuron_Name(convolutiondepthwise2);
			Declear_Opration(prelu, relu4);
			Neuron_Name(relu4);
			Declear_Opration(baseconv, convolution4);
			Neuron_Name(convolution4);
			Declear_Opration(eltwise, fuse3);
			Neuron_Name(fuse3);
			Declear_Opration(baseconv, convolution5);
			Neuron_Name(convolution5);
			Declear_Opration(prelu, relu5);
			Neuron_Name(relu5);
			Declear_Opration(baseconv, convolutiondepthwise3);
			Neuron_Name(convolutiondepthwise3);
			Declear_Opration(prelu, relu6);
			Neuron_Name(relu6);
			Declear_Opration(baseconv, convolution6);
			Neuron_Name(convolution6);
			Declear_Opration(baseconv, convolution7);
			Neuron_Name(convolution7);
			Declear_Opration(prelu, relu7);
			Neuron_Name(relu7);
			Declear_Opration(baseconv, convolutiondepthwise4);
			Neuron_Name(convolutiondepthwise4);
			Declear_Opration(prelu, relu8);
			Neuron_Name(relu8);
			Declear_Opration(baseconv, convolution8);
			Neuron_Name(convolution8);
			Declear_Opration(eltwise, fuse5);
			Neuron_Name(fuse5);
			Declear_Opration(baseconv, convolution9);
			Neuron_Name(convolution9);
			Declear_Opration(prelu, relu9);
			Neuron_Name(relu9);
			Declear_Opration(baseconv, convolutiondepthwise5);
			Neuron_Name(convolutiondepthwise5);
			Declear_Opration(prelu, relu10);
			Neuron_Name(relu10);
			Declear_Opration(baseconv, convolution10);
			Neuron_Name(convolution10);
			Declear_Opration(baseconv, convolution11);
			Neuron_Name(convolution11);
			Declear_Opration(prelu, relu11);
			Neuron_Name(relu11);
			Declear_Opration(baseconv, convolutiondepthwise6);
			Neuron_Name(convolutiondepthwise6);
			Declear_Opration(prelu, relu12);
			Neuron_Name(relu12);
			Declear_Opration(baseconv, convolution12);
			Neuron_Name(convolution12);
			Declear_Opration(eltwise, fuse7);
			Neuron_Name(fuse7);
			Declear_Opration(inner_product, fc1);
			Neuron_Name(fc1);
			Declear_Opration(prelu, fc1_relu);
			Neuron_Name(fc1_relu);
			Declear_Opration(inner_product, fc2);
			Neuron_Name(fc2);
			Declear_Opration(prelu, fc2_relu);
			Neuron_Name(fc2_relu);
			Declear_Opration(inner_product, fc3);
			Neuron_Name(fc3);
			Declear_Opration(inner_product, fc4);
			Neuron_Name(fc4);


#ifdef USE_CUDA
			cublasHandle_t cublas_handle_ = nullptr;
			void Forward_gpu_native(const std::shared_ptr<tensor<float>> input_data);
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle_ = nullptr;
			void Forward_gpu_cudnn(const std::shared_ptr<tensor<float>> input_data);
#endif 
#endif
			void Forward_cpu(const std::shared_ptr<tensor<float>> input_data);

		public:
			Hera(int device);
			~Hera();

			std::vector<float> Forward(const unsigned char* input_data, int num, int order = 0)
			{
				if (order == 0)//NCHW
				{
					tensor_unsigned_char_data.reset(new tensor<unsigned char>(std::vector<int>{(int)num, 3, 64, 64}, device_, NCHW));
					tensor_float_data.reset(new tensor<float>(std::vector<int>{(int)num, 3, 64, 64}, device_, NCHW));
				}
				else//NHWC
				{
					tensor_unsigned_char_data.reset(new tensor<unsigned char>(std::vector<int>{(int)num, 64, 64, 3}, device_, NHWC));
					tensor_float_data.reset(new tensor<float>(std::vector<int>{(int)num, 64, 64, 3}, device_, NHWC));
				}

				float means[3] = { 127.5f, 127.5f, 127.5f };
				float var = 0.0078125;
				if (device_ < 0)
				{
					unsigned char* tensor_data = tensor_unsigned_char_data->mutable_cpu_data();
					memcpy(tensor_data, input_data, num * 3 * 64 * 64 * sizeof(unsigned char));
					tensor_operation_cpu::preprocess_tensors_cpu(tensor_unsigned_char_data, tensor_float_data, means, var);
					Forward_cpu(tensor_float_data);
				}
				else
				{
#ifdef USE_CUDA
					unsigned char* tensor_data = tensor_unsigned_char_data->mutable_gpu_data();
					CUDA_CHECK(cudaMemcpy(tensor_data, input_data, num * 3 * 64 * 64 * sizeof(unsigned char), cudaMemcpyDefault));
					tensor_operation_gpu::preprocess_tensors_gpu(tensor_unsigned_char_data, tensor_float_data, means, var);
#ifdef USE_CUDNN
					Forward_gpu_cudnn(tensor_float_data);
#else
					Forward_gpu_native(tensor_float_data);
#endif //!USE_CUDNN

#else
					NO_GPU;
#endif//!USE_CUDA
				}

				const float* conv_fc4_data = get_fc4()->cpu_data();
				std::vector<float> result(136);
				memcpy(result.data(), conv_fc4_data, 136 * sizeof(float));

				return result;
			}
		};
	}
}

#endif // !_HERA_HPP_