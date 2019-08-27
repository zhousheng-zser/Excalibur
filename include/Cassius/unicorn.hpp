#ifndef _UNICORN_HPP_
#define _UNICORN_HPP_

#include "../Excalibur/support_layers.hpp"
#include "../Excalibur/tensor_operation_cpu.hpp"
#include "../Excalibur/tensor_operation_gpu.hpp"
#include "vunicorn.hpp"

using namespace glasssix::excalibur;

namespace glasssix
{
	namespace cassius
	{
		class Unicorn: public vUnicorn
		{
			Declear_Params(conv1a);
			Declear_Params(relu1a);
			Declear_Params(conv1b);
			Declear_Params(relu1b);
			Declear_Params(conv2_1);
			Declear_Params(relu2_1);
			Declear_Params(conv2_2);
			Declear_Params(relu2_2);
			Declear_Params(conv2);
			Declear_Params(relu2);
			Declear_Params(conv3_1);
			Declear_Params(relu3_1);
			Declear_Params(conv3_2);
			Declear_Params(relu3_2);
			Declear_Params(conv3_3);
			Declear_Params(relu3_3);
			Declear_Params(conv3_4);
			Declear_Params(relu3_4);
			Declear_Params(conv3);
			Declear_Params(relu3);
			Declear_Params(conv4_1);
			Declear_Params(relu4_1);
			Declear_Params(conv4_2);
			Declear_Params(relu4_2);
			Declear_Params(conv4_3);
			Declear_Params(relu4_3);
			Declear_Params(conv4_4);
			Declear_Params(relu4_4);
			Declear_Params(conv4_5);
			Declear_Params(relu4_5);
			Declear_Params(conv4_6);
			Declear_Params(relu4_6);
			Declear_Params(conv4_7);
			Declear_Params(relu4_7);
			Declear_Params(conv4_8);
			Declear_Params(relu4_8);
			Declear_Params(conv4_9);
			Declear_Params(relu4_9);
			Declear_Params(conv4_10);
			Declear_Params(relu4_10);
			Declear_Params(conv4);
			Declear_Params(relu4);
			Declear_Params(conv5_1);
			Declear_Params(relu5_1);
			Declear_Params(conv5_2);
			Declear_Params(relu5_2);
			Declear_Params(conv5_3);
			Declear_Params(relu5_3);
			Declear_Params(conv5_4);
			Declear_Params(relu5_4);
			Declear_Params(conv5_5);
			Declear_Params(relu5_5);
			Declear_Params(conv5_6);
			Declear_Params(relu5_6);
			Declear_Params(conv5);
			Declear_Params(relu5);

			//
			int device_;
			bool cudnn_ready_ = false;
			bool int8_quantization_ = false;

			std::shared_ptr<tensor<unsigned char>> tensor_unsigned_char_data = nullptr;
			std::shared_ptr<tensor<float>> tensor_float_data = nullptr;
			std::vector<float> quality_score;
			//

			Declear_Opration(flip, fliper);
			Neuron_Name(flip);
			Declear_Opration(concat, concator);
			Neuron_Name(concat);
			Declear_Opration(baseconv, conv1a);
			Neuron_Name(conv1a);
			Declear_Opration(prelu, relu1a);
			Neuron_Name(relu1a);
			Declear_Opration(baseconv, conv1b);
			Neuron_Name(conv1b);
			Declear_Opration(prelu, relu1b);
			Neuron_Name(relu1b);
			Declear_Opration(pooling, pool1b);
			Neuron_Name(pool1b);
			Declear_Opration(baseconv, conv2_1);
			Neuron_Name(conv2_1);
			Declear_Opration(prelu, relu2_1);
			Neuron_Name(relu2_1);
			Declear_Opration(baseconv, conv2_2);
			Neuron_Name(conv2_2);
			Declear_Opration(prelu, relu2_2);
			Neuron_Name(relu2_2);
			Declear_Opration(eltwise, res2_2);
			Neuron_Name(res2_2);
			Declear_Opration(baseconv, conv2);
			Neuron_Name(conv2);
			Declear_Opration(prelu, relu2);
			Neuron_Name(relu2);
			Declear_Opration(pooling, pool2);
			Neuron_Name(pool2);
			Declear_Opration(baseconv, conv3_1);
			Neuron_Name(conv3_1);
			Declear_Opration(prelu, relu3_1);
			Neuron_Name(relu3_1);
			Declear_Opration(baseconv, conv3_2);
			Neuron_Name(conv3_2);
			Declear_Opration(prelu, relu3_2);
			Neuron_Name(relu3_2);
			Declear_Opration(eltwise, res3_2);
			Neuron_Name(res3_2);
			Declear_Opration(baseconv, conv3_3);
			Neuron_Name(conv3_3);
			Declear_Opration(prelu, relu3_3);
			Neuron_Name(relu3_3);
			Declear_Opration(baseconv, conv3_4);
			Neuron_Name(conv3_4);
			Declear_Opration(prelu, relu3_4);
			Neuron_Name(relu3_4);
			Declear_Opration(eltwise, res3_4);
			Neuron_Name(res3_4);
			Declear_Opration(baseconv, conv3);
			Neuron_Name(conv3);
			Declear_Opration(prelu, relu3);
			Neuron_Name(relu3);
			Declear_Opration(pooling, pool3);
			Neuron_Name(pool3);
			Declear_Opration(baseconv, conv4_1);
			Neuron_Name(conv4_1);
			Declear_Opration(prelu, relu4_1);
			Neuron_Name(relu4_1);
			Declear_Opration(baseconv, conv4_2);
			Neuron_Name(conv4_2);
			Declear_Opration(prelu, relu4_2);
			Neuron_Name(relu4_2);
			Declear_Opration(eltwise, res4_2);
			Neuron_Name(res4_2);
			Declear_Opration(baseconv, conv4_3);
			Neuron_Name(conv4_3);
			Declear_Opration(prelu, relu4_3);
			Neuron_Name(relu4_3);
			Declear_Opration(baseconv, conv4_4);
			Neuron_Name(conv4_4);
			Declear_Opration(prelu, relu4_4);
			Neuron_Name(relu4_4);
			Declear_Opration(eltwise, res4_4);
			Neuron_Name(res4_4);
			Declear_Opration(baseconv, conv4_5);
			Neuron_Name(conv4_5);
			Declear_Opration(prelu, relu4_5);
			Neuron_Name(relu4_5);
			Declear_Opration(baseconv, conv4_6);
			Neuron_Name(conv4_6);
			Declear_Opration(prelu, relu4_6);
			Neuron_Name(relu4_6);
			Declear_Opration(eltwise, res4_6);
			Neuron_Name(res4_6);
			Declear_Opration(baseconv, conv4_7);
			Neuron_Name(conv4_7);
			Declear_Opration(prelu, relu4_7);
			Neuron_Name(relu4_7);
			Declear_Opration(baseconv, conv4_8);
			Neuron_Name(conv4_8);
			Declear_Opration(prelu, relu4_8);
			Neuron_Name(relu4_8);
			Declear_Opration(eltwise, res4_8);
			Neuron_Name(res4_8);
			Declear_Opration(baseconv, conv4_9);
			Neuron_Name(conv4_9);
			Declear_Opration(prelu, relu4_9);
			Neuron_Name(relu4_9);
			Declear_Opration(baseconv, conv4_10);
			Neuron_Name(conv4_10);
			Declear_Opration(prelu, relu4_10);
			Neuron_Name(relu4_10);
			Declear_Opration(eltwise, res4_10);
			Neuron_Name(res4_10);
			Declear_Opration(baseconv, conv4);
			Neuron_Name(conv4);
			Declear_Opration(prelu, relu4);
			Neuron_Name(relu4);
			Declear_Opration(pooling, pool4);
			Neuron_Name(pool4);
			Declear_Opration(baseconv, conv5_1);
			Neuron_Name(conv5_1);
			Declear_Opration(prelu, relu5_1);
			Neuron_Name(relu5_1);
			Declear_Opration(baseconv, conv5_2);
			Neuron_Name(conv5_2);
			Declear_Opration(prelu, relu5_2);
			Neuron_Name(relu5_2);
			Declear_Opration(eltwise, res5_2);
			Neuron_Name(res5_2);
			Declear_Opration(baseconv, conv5_3);
			Neuron_Name(conv5_3);
			Declear_Opration(prelu, relu5_3);
			Neuron_Name(relu5_3);
			Declear_Opration(baseconv, conv5_4);
			Neuron_Name(conv5_4);
			Declear_Opration(prelu, relu5_4);
			Neuron_Name(relu5_4);
			Declear_Opration(eltwise, res5_4);
			Neuron_Name(res5_4);
			Declear_Opration(baseconv, conv5_5);
			Neuron_Name(conv5_5);
			Declear_Opration(prelu, relu5_5);
			Neuron_Name(relu5_5);
			Declear_Opration(baseconv, conv5_6);
			Neuron_Name(conv5_6);
			Declear_Opration(prelu, relu5_6);
			Neuron_Name(relu5_6);
			Declear_Opration(eltwise, res5_6);
			Neuron_Name(res5_6);
			Declear_Opration(baseconv, conv5);
			Neuron_Name(conv5);
			Declear_Opration(prelu, relu5);
			Neuron_Name(relu5);
			Declear_Opration(pooling, pool5);
			Neuron_Name(pool5);
			Declear_Opration(mirrormax, mirrmax);
			Neuron_Name(feature);
			Declear_Opration(normalize, normalizer);

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
			Unicorn(int device);
			virtual ~Unicorn();

			std::vector<std::vector<float> > Forward(const float* input_data, unsigned num, int order = 0) override;

			std::vector<std::vector<float> > Forward(const unsigned char* input_data, unsigned num, int order = 0) override;


			static int get_input_channel()
			{
				return 3;
			}

			static int get_input_width()
			{
				return 128;
			}

			static int get_input_height()
			{
				return 128;
			}

		};
	}
}

#endif //!_UNICORN_HPP_
