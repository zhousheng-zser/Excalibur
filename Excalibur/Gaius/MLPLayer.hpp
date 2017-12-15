#pragma once
#ifndef _MLPLAYER_HPP_
#define _MLPLAYER_HPP_
#include <vector>
#include "math_helper.hpp"

namespace excalibur
{
	class MLPLayer {
	public:
		explicit MLPLayer(int32_t act_func_type = 1)
			: input_dim_(0), output_dim_(0), act_func_type_(act_func_type) {}
		~MLPLayer() {}

		void Compute(const float* input, float* output)
		{
#pragma omp parallel num_threads(SEETA_NUM_THREADS)
			{
#pragma omp for nowait
				for (int32_t i = 0; i < output_dim_; i++) {
					output[i] = MathHelper::VectorInnerProductCPU(input,
						weights_.data() + i * input_dim_, input_dim_) + bias_[i];
					output[i] = (act_func_type_ == 1 ? ReLU(output[i]) : Sigmoid(-output[i]));
				}
			}
		}

		inline int32_t GetInputDim() const { return input_dim_; }
		inline int32_t GetOutputDim() const { return output_dim_; }

		inline void SetSize(int32_t inputDim, int32_t outputDim) {
			if (inputDim <= 0 || outputDim <= 0) {
				return;  // @todo handle the errors!!!
			}
			input_dim_ = inputDim;
			output_dim_ = outputDim;
			weights_.resize(inputDim * outputDim);
			bias_.resize(outputDim);
		}

		inline void SetWeights(const float* weights, int32_t len) {
			if (weights == nullptr || len != input_dim_ * output_dim_) {
				return;  // @todo handle the errors!!!
			}
			std::copy(weights, weights + input_dim_ * output_dim_, weights_.begin());
		}

		inline void SetBias(const float* bias, int32_t len) {
			if (bias == nullptr || len != output_dim_) {
				return;  // @todo handle the errors!!!
			}
			std::copy(bias, bias + output_dim_, bias_.begin());
		}

	private:
		inline float Sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(x));
		}

		inline float ReLU(float x) {
			return (x > 0.0f ? x : 0.0f);
		}

	private:
		int32_t act_func_type_;
		int32_t input_dim_;
		int32_t output_dim_;
		std::vector<float> weights_;
		std::vector<float> bias_;
	};
}

#endif // _MLPLAYER_HPP_