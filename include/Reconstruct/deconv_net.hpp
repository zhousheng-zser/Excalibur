#ifndef _DECONV_NET_HPP_
#define _DECONV_NET_HPP_
#include <vector>
#include "../Excalibur/support_layers.hpp"
#include "../Excalibur/tensor_operation_cpu.hpp"
#include "../Excalibur/tensor_operation_gpu.hpp"

using namespace glasssix::excalibur;

namespace glasssix
{
	namespace longinus
	{
		class deconvT
		{
			Declear_Params(conv1);
			Declear_Params(relu1);
			Declear_Params(conv2);
			Declear_Params(relu2);
			Declear_Params(conv22);
			Declear_Params(relu22);
			Declear_Params(conv23);
			Declear_Params(relu23);
			Declear_Params(conv24);
			Declear_Params(relu24);
			Declear_Params(conv25);
			Declear_Params(relu25);
			Declear_Params(conv26);
			Declear_Params(relu26);
			Declear_Params(conv3);

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
			Declear_Opration(baseconv, conv2);
			Neuron_Name(conv2);
			Declear_Opration(prelu, relu2);
			Neuron_Name(relu2);
			Declear_Opration(baseconv, conv22);
			Neuron_Name(conv22);
			Declear_Opration(prelu, relu22);
			Neuron_Name(relu22);
			Declear_Opration(baseconv, conv23);
			Neuron_Name(conv23);
			Declear_Opration(prelu, relu23);
			Neuron_Name(relu23);
			Declear_Opration(eltwise, res2_3);
			Neuron_Name(res2_3);
			Declear_Opration(baseconv, conv24);
			Neuron_Name(conv24);
			Declear_Opration(prelu, relu24);
			Neuron_Name(relu24);
			Declear_Opration(baseconv, conv25);
			Neuron_Name(conv25);
			Declear_Opration(prelu, relu25);
			Neuron_Name(relu25);
			Declear_Opration(eltwise, res2_5);
			Neuron_Name(res2_5);
			Declear_Opration(baseconv, conv26);
			Neuron_Name(conv26);
			Declear_Opration(prelu, relu26);
			Neuron_Name(relu26);
			Declear_Opration(deconv, conv3);
			Neuron_Name(conv3);

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
			deconvT(int device);
			virtual ~deconvT();

			std::vector<std::vector<unsigned char> > Forward(const unsigned char* input_data, unsigned num, int order = 0)
			{
				std::vector<std::vector<unsigned char> > feature;

				if (order == 0)//NCHW
				{
					tensor_unsigned_char_data.reset(new tensor<unsigned char>(std::vector<int>{(int)num, 1, 126, 102}, device_, NCHW));
					tensor_float_data.reset(new tensor<float>(std::vector<int>{(int)num, 1, 126, 102}, device_, NCHW));
				}
				else//NHWC
				{
					tensor_unsigned_char_data.reset(new tensor<unsigned char>(std::vector<int>{(int)num, 126, 102, 1}, device_, NHWC));
					tensor_float_data.reset(new tensor<float>(std::vector<int>{(int)num, 126, 102, 1}, device_, NHWC));
				}
				
				if (device_<0)
				{
					unsigned char* tensor_data = tensor_unsigned_char_data->mutable_cpu_data();
					memcpy(tensor_data, input_data, num * 1 * 126 * 102 * sizeof(unsigned char));
					tensor_operation_cpu::preprocess_tensors_cpu(tensor_unsigned_char_data, tensor_float_data);
					Forward_cpu(tensor_float_data);
					std::vector<float> temp(364 * 292);
					std::vector<unsigned char> temp_ch(364 * 292);
				
					std::memcpy(temp.data(), get_conv3()->cpu_data(), 364 * 292 * sizeof(float));
					for (int i = 0; i < 364 * 292; i++)
					{
						temp[i] = temp[i] / 0.0078125 + 127.5;
						if (temp[i] > 255)
						{
							temp_ch[i] = (unsigned char)255;
						}
						else if (temp[i] < 0)
						{
							temp_ch[i] = (unsigned char)0;
						}
						else
						{
							temp_ch[i] = (unsigned char)temp[i];
						}
					}
					feature.push_back(temp_ch);
					return feature;
				}
				else
				{
#ifdef USE_CUDA
					unsigned char* tensor_data = tensor_unsigned_char_data->mutable_gpu_data();
					cudaMemcpy(tensor_data, input_data, num * 1 * 48 * 48 * sizeof(unsigned char), cudaMemcpyDefault);
					tensor_operation_gpu::preprocess_tensors_gpu(tensor_unsigned_char_data, tensor_float_data);
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

#endif // !_DECONV_NET_HPP_