#pragma once
#ifndef _OPERATION_LSTM_HPP_
#define _OPERATION_LSTM_HPP_
#include "operation.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		class operation_lstm : public operation<Dtype>
		{
		public:
			explicit operation_lstm(const operation_param& param);

			virtual const char* type() const { return this->params_.type_.c_str(); }

			virtual ~operation_lstm() {}

			virtual int init_weights();

			virtual int init_weights(FILE* fp);

		protected:
			virtual void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
				std::vector<std::shared_ptr<memory::tensor<float>>>& tops);

			virtual void lstm_cpu_f32(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top, int reverse);

			virtual void forward_gpu_f32(
#ifdef USE_CUDA
				cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
				cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
				const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
				std::vector<std::shared_ptr<memory::tensor<float>>>& tops);

		private:
			std::shared_ptr<memory::tensor<float>> hidden_;
			std::shared_ptr<memory::tensor<float>> cell_;
			std::shared_ptr<memory::tensor<float>> gates_;

			int num_output_ = 0;
			int weight_data_size_ = 0;
			int direction_ = 1;
		};
	}
}
#endif // !_OPERATION_LSTM_HPP_

