#ifndef _Lindburg_HPP_
#define _Lindburg_HPP_

#include "../Excalibur/support_layers.hpp"
#include "../Excalibur/tensor_operation_cpu.hpp"
#include "../Excalibur/tensor_operation_gpu.hpp"

using namespace glasssix::excalibur;

namespace glasssix
{
	namespace longinus
	{
		class Banshee
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
			Declear_Params(conv6_2);
			Declear_Params(conv6_3);

			//
			int device_;
			bool cudnn_ready_ = false;
			bool int8_quantization_ = false;
			std::shared_ptr<tensor<float>> tensor_float_data = nullptr;
			std::shared_ptr<tensor<unsigned char>> tensor_unsigned_char_data = nullptr;
			//

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
			Declear_Opration(sigmoid, sigmoid1);
			Neuron_Name(sigmoid1);
			Declear_Opration(inner_product, conv6_2);
			Neuron_Name(conv6_2);
			Declear_Opration(inner_product, conv6_3);
			Neuron_Name(conv6_3);

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
			Banshee(int device);
			virtual ~Banshee();

			void Forward(const float* input_data, int num, int order = 0)
			{
				if (order == 0)//NCHW
				{
					tensor_float_data.reset(new tensor<float>(std::vector<int>{(int)num, 1, 48, 48}, device_, NCHW));
				}
				else//NHWC
				{
					tensor_float_data.reset(new tensor<float>(std::vector<int>{(int)num, 48, 48, 1}, device_, NHWC));
				}

				float means[3] = { 104.0f, 117.0f, 124.0f };
				if (device_<0)
				{
					float* tensor_data = tensor_float_data->mutable_cpu_data();
					memcpy(tensor_data, input_data, num * 1 * 48 * 48 * sizeof(float));
					tensor_operation_cpu::preprocess_tensors_cpu(tensor_float_data, tensor_float_data, means);

					std::shared_ptr<tensor<float>> src_tensor = tensor_float_data;
#ifdef __ARM_NEON
					if (order == 1)
						tensor_operation_cpu::nhwc2nchw_cpu(tensor_float_data, src_tensor);
#endif
					Forward_cpu(src_tensor);
				}
				else
				{
#ifdef USE_CUDA
					float* tensor_data = tensor_float_data->mutable_gpu_data();
					CUDA_CHECK(cudaMemcpy(tensor_data, input_data, num * 1 * 48 * 48 * sizeof(float), cudaMemcpyDefault));
					tensor_operation_gpu::preprocess_tensors_gpu(tensor_float_data, tensor_float_data, means);
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

			void Forward(const unsigned char* input_data, int num, int order = 0)
			{
				if (order == 0)//NCHW
				{
					tensor_unsigned_char_data.reset(new tensor<unsigned char>(std::vector<int>{(int)num, 1, 48, 48}, device_, NCHW));
					tensor_float_data.reset(new tensor<float>(std::vector<int>{(int)num, 1, 48, 48}, device_, NCHW));
				}
				else//NHWC
				{
					tensor_unsigned_char_data.reset(new tensor<unsigned char>(std::vector<int>{(int)num, 48, 48, 1}, device_, NHWC));
					tensor_float_data.reset(new tensor<float>(std::vector<int>{(int)num, 48, 48, 1}, device_, NHWC));
				}

				float means[3] = { 104.0f, 117.0f, 124.0f };
				if (device_<0)
				{
					unsigned char* tensor_data = tensor_unsigned_char_data->mutable_cpu_data();
					memcpy(tensor_data, input_data, num * 1 * 48 * 48 * sizeof(unsigned char));
					tensor_operation_cpu::preprocess_tensors_cpu(tensor_unsigned_char_data, tensor_float_data, means);
					Forward_cpu(tensor_float_data);
				}
				else
				{
#ifdef USE_CUDA
					unsigned char* tensor_data = tensor_unsigned_char_data->mutable_gpu_data();
					CUDA_CHECK(cudaMemcpy(tensor_data, input_data, num * 1 * 48 * 48 * sizeof(unsigned char), cudaMemcpyDefault));
					tensor_operation_gpu::preprocess_tensors_gpu(tensor_unsigned_char_data, tensor_float_data, means);
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

			void getParam(std::vector<std::vector<float> > &keypointParam, int num);

			std::vector<unsigned char> alignFace(const unsigned char* ori_image, int n, int channels, int height, int width, 
				std::vector<std::vector<int>> bbox, std::vector<std::vector<int> >landmarks);

			std::vector<unsigned char> alignFace(const unsigned char* ori_image, int n, int channels, int height, int width);

		};
	}
}

#endif // !_Lindburg_HPP_