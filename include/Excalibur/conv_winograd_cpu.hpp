#ifndef _CONV_WINOGRAD_CPU_HPP_
#define _CONV_WINOGRAD_CPU_HPP_
#include "base_conv.hpp"

namespace glasssix
{
	namespace excalibur
	{
		class conv_winograd_cpu : public baseconv
		{
		public:
			int m_ = 2;
			int m_length_;
			int kernel_length_;
			int tile_size_;
			int tile_length_;
			int h_tile_num_;
			int w_tile_num_;
			int V_num_;//the quantity of V
			int U_num_;//the quantity of U

			std::shared_ptr<tensor<float>> U_, V_;
			float *U_data, *V_data;

			std::shared_ptr<tensor<short>> U_int16, V_int16;
			short *U_int16_data, *V_int16_data;

			conv_winograd_cpu(int input_Channel, int output_Channel, int group, int kernelSize, int stride, int pad, bool bias_term, int device = -1, bool int8_quantization = false);

			virtual ~conv_winograd_cpu() {};

			void Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top) override;

		private:

			void forward_gemm(const float* input, const float* weights, float* output, bool skip_im2col = false) override;

			void forward_gemm(const signed char* input, const signed char* weights, int* output, bool skip_im2col = false) override;

			void forward_bias(float* output, const float* bias) override;
			

			inline bool is_a_ge_zero_and_a_lt_b(int a, int b)
			{
				return static_cast<unsigned>(a) < static_cast<unsigned>(b);
			}

			//fp32
			inline void calculate_GgGT(const float *weight_data, float *u_data)
			{
				u_data[0] = weight_data[0];
				u_data[1] = (weight_data[0] + weight_data[1] + weight_data[2]) / 2;
				u_data[2] = (weight_data[0] - weight_data[1] + weight_data[2]) / 2;
				u_data[3] = weight_data[2];
				u_data[4] = (weight_data[0] + weight_data[3] + weight_data[6]) / 2;
				u_data[5] = (weight_data[0] + weight_data[1] + weight_data[2] +
					weight_data[3] + weight_data[4] + weight_data[5] +
					weight_data[6] + weight_data[7] + weight_data[8]) / 4;
				u_data[6] = (weight_data[0] - weight_data[1] + weight_data[2] +
					weight_data[3] - weight_data[4] + weight_data[5] +
					weight_data[6] - weight_data[7] + weight_data[8]) / 4;
				u_data[7] = (weight_data[2] + weight_data[5] + weight_data[8]) / 2;
				u_data[8] = (weight_data[0] - weight_data[3] + weight_data[6]) / 2;
				u_data[9] = (weight_data[0] + weight_data[1] + weight_data[2] -
					weight_data[3] - weight_data[4] - weight_data[5] +
					weight_data[6] + weight_data[7] + weight_data[8]) / 4;
				u_data[10] = (weight_data[0] - weight_data[1] + weight_data[2] -
					weight_data[3] + weight_data[4] - weight_data[5] +
					weight_data[6] - weight_data[7] + weight_data[8]) / 4;
				u_data[11] = (weight_data[2] - weight_data[5] + weight_data[8]) / 2;
				u_data[12] = weight_data[6];
				u_data[13] = (weight_data[6] + weight_data[7] + weight_data[8]) / 2;
				u_data[14] = (weight_data[6] - weight_data[7] + weight_data[8]) / 2;
				u_data[15] = weight_data[8];
			}

			inline void calculate_BTdB(const float *tile_data, float *v_data)
			{
				v_data[0] = tile_data[0] - tile_data[2] - tile_data[8] + tile_data[10];
				v_data[1] = tile_data[1] + tile_data[2] - tile_data[9] - tile_data[10];
				v_data[2] = -tile_data[1] + tile_data[2] + tile_data[9] - tile_data[10];
				v_data[3] = tile_data[1] - tile_data[3] - tile_data[9] + tile_data[11];
				v_data[4] = tile_data[4] - tile_data[6] + tile_data[8] - tile_data[10];
				v_data[5] = tile_data[5] + tile_data[6] + tile_data[9] + tile_data[10];
				v_data[6] = -tile_data[5] + tile_data[6] - tile_data[9] + tile_data[10];
				v_data[7] = tile_data[5] - tile_data[7] + tile_data[9] - tile_data[11];
				v_data[8] = -tile_data[4] + tile_data[6] + tile_data[8] - tile_data[10];
				v_data[9] = -tile_data[5] - tile_data[6] + tile_data[9] + tile_data[10];
				v_data[10] = tile_data[5] - tile_data[6] - tile_data[9] + tile_data[10];
				v_data[11] = -tile_data[5] + tile_data[7] + tile_data[9] - tile_data[11];
				v_data[12] = tile_data[4] - tile_data[6] - tile_data[12] + tile_data[14];
				v_data[13] = tile_data[5] + tile_data[6] - tile_data[13] - tile_data[14];
				v_data[14] = -tile_data[5] + tile_data[6] + tile_data[13] - tile_data[14];
				v_data[15] = tile_data[5] - tile_data[7] - tile_data[13] + tile_data[15];
			}

			inline void calculate_ATmA(const float *m_data, float *result)
			{
				result[0] = m_data[0] + m_data[1] + m_data[2] + m_data[4] + m_data[5] + m_data[6] + m_data[8] + m_data[9] + m_data[10];
				result[1] = m_data[1] - m_data[2] - m_data[3] + m_data[5] - m_data[6] - m_data[7] + m_data[9] - m_data[10] - m_data[11];
				result[2] = m_data[4] + m_data[5] + m_data[6] - m_data[8] - m_data[9] - m_data[10] - m_data[12] - m_data[13] - m_data[14];
				result[3] = m_data[5] - m_data[6] - m_data[7] - m_data[9] + m_data[10] + m_data[11] - m_data[13] + m_data[14] + m_data[15];
			}

			//int8
			inline void calculate_GgGT(const signed char *weight_data, short *u_data)
			{
				//mutiply 4 to avoid accuracy loss
				u_data[0] = weight_data[0] * 4;
				u_data[1] = (short(weight_data[0]) + short(weight_data[1]) + short(weight_data[2])) * 2;
				u_data[2] = (short(weight_data[0]) - short(weight_data[1]) + short(weight_data[2])) * 2;
				u_data[3] = short(weight_data[2]) * 4;
				u_data[4] = (short(weight_data[0]) + short(weight_data[3]) + short(weight_data[6])) * 2;
				u_data[5] = (short(weight_data[0]) + short(weight_data[1]) + short(weight_data[2]) +
					short(weight_data[3]) + short(weight_data[4]) + short(weight_data[5]) +
					short(weight_data[6]) + short(weight_data[7]) + short(weight_data[8]));
				u_data[6] = (short(weight_data[0]) - short(weight_data[1]) + short(weight_data[2]) +
					short(weight_data[3]) - short(weight_data[4]) + short(weight_data[5]) +
					short(weight_data[6]) - short(weight_data[7]) + short(weight_data[8]));
				u_data[7] = (short(weight_data[2]) + short(weight_data[5]) + short(weight_data[8])) * 2;
				u_data[8] = (short(weight_data[0]) - short(weight_data[3]) + short(weight_data[6])) * 2;
				u_data[9] = (short(weight_data[0]) + short(weight_data[1]) + short(weight_data[2]) -
					short(weight_data[3]) - short(weight_data[4]) - short(weight_data[5]) +
					short(weight_data[6]) + short(weight_data[7]) + short(weight_data[8]));
				u_data[10] = (short(weight_data[0]) - short(weight_data[1]) + short(weight_data[2]) -
					short(weight_data[3]) + short(weight_data[4]) - short(weight_data[5]) +
					short(weight_data[6]) - short(weight_data[7]) + short(weight_data[8]));
				u_data[11] = (short(weight_data[2]) - short(weight_data[5]) + short(weight_data[8])) * 2;
				u_data[12] = weight_data[6] * 4;
				u_data[13] = (short(weight_data[6]) + short(weight_data[7]) + short(weight_data[8])) * 2;
				u_data[14] = (short(weight_data[6]) - short(weight_data[7]) + short(weight_data[8])) * 2;
				u_data[15] = weight_data[8] * 4;
			}

			inline void calculate_BTdB(const signed char *tile_data, short *v_data)
			{
				v_data[0] = short(tile_data[0]) - short(tile_data[2]) - short(tile_data[8]) + short(tile_data[10]);
				v_data[1] = short(tile_data[1]) + short(tile_data[2]) - short(tile_data[9]) - short(tile_data[10]);
				v_data[2] = -short(tile_data[1]) + short(tile_data[2]) + short(tile_data[9]) - short(tile_data[10]);
				v_data[3] = short(tile_data[1]) - short(tile_data[3]) - short(tile_data[9]) + short(tile_data[11]);
				v_data[4] = short(tile_data[4]) - short(tile_data[6]) + short(tile_data[8]) - short(tile_data[10]);
				v_data[5] = short(tile_data[5]) + short(tile_data[6]) + short(tile_data[9]) + short(tile_data[10]);
				v_data[6] = -short(tile_data[5]) + short(tile_data[6]) - short(tile_data[9]) + short(tile_data[10]);
				v_data[7] = short(tile_data[5]) - short(tile_data[7]) + short(tile_data[9]) - short(tile_data[11]);
				v_data[8] = -short(tile_data[4]) + short(tile_data[6]) + short(tile_data[8]) - short(tile_data[10]);
				v_data[9] = -short(tile_data[5]) - short(tile_data[6]) + short(tile_data[9]) + short(tile_data[10]);
				v_data[10] = short(tile_data[5]) - short(tile_data[6]) - short(tile_data[9]) + short(tile_data[10]);
				v_data[11] = -short(tile_data[5]) + short(tile_data[7]) + short(tile_data[9]) - short(tile_data[11]);
				v_data[12] = short(tile_data[4]) - short(tile_data[6]) - short(tile_data[12]) + short(tile_data[14]);
				v_data[13] = short(tile_data[5]) + short(tile_data[6]) - short(tile_data[13]) - short(tile_data[14]);
				v_data[14] = -short(tile_data[5]) + short(tile_data[6]) + short(tile_data[13]) - short(tile_data[14]);
				v_data[15] = short(tile_data[5]) - short(tile_data[7]) - short(tile_data[13]) + short(tile_data[15]);
			}

			inline void calculate_ATmA(const int *m_data, int *result)
			{
				result[0] = m_data[0] + m_data[1] + m_data[2] + m_data[4] + m_data[5] + m_data[6] + m_data[8] + m_data[9] + m_data[10];
				result[1] = m_data[1] - m_data[2] - m_data[3] + m_data[5] - m_data[6] - m_data[7] + m_data[9] - m_data[10] - m_data[11];
				result[2] = m_data[4] + m_data[5] + m_data[6] - m_data[8] - m_data[9] - m_data[10] - m_data[12] - m_data[13] - m_data[14];
				result[3] = m_data[5] - m_data[6] - m_data[7] - m_data[9] + m_data[10] + m_data[11] - m_data[13] + m_data[14] + m_data[15];
			}


#ifdef USE_CUDA
			void Forward(cublasHandle_t cublas_handle_, const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top) {}
			void forward_gemm(cublasHandle_t cublas_handle_, const float* input, const float* weights, float* output, bool skip_im2col = false) {}
			void forward_gemm(cublasHandle_t cublas_handle_, const signed char* input, const signed char* weights, int* output, bool skip_im2col = false) {}
			void forward_bias(cublasHandle_t cublas_handle_, float* output, const float* bias) {}
#ifdef USE_CUDNN
			void Forward(cudnnHandle_t cudnn_handle_, const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top) {}
#endif //!USE_CUDNN
#endif // !USE_CUDA
		};
	}
}


#endif // !_CONV_WINOGRAD_CPU_HPP_