#include "../../include/Excalibur/operation_lstm.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"
#include "../../include/Excalibur/math_functions.hpp"
#include <random>

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_lstm<Dtype>::operation_lstm(const operation_param& param) : operation<Dtype>(param)
		{
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					num_output_ = atoi(split_string(attrs[i], "=")[1].c_str());

					hidden_.reset(new memory::tensor<float>(num_output_, this->params_.device_, memory::NCHW));
					cell_.reset(new memory::tensor<float>(num_output_, this->params_.device_, memory::NCHW));
					gates_.reset(new memory::tensor<float>(num_output_, 4, this->params_.device_, memory::NCHW));
				}
				else if (split_string(attrs[i], "=")[0] == "1")
				{
					weight_data_size_ = (bool)atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "2")
				{
					direction_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "-23330")
				{
					//do nothing
				}
				else
				{
					LOG(FATAL) << "Un-supported InnerProduct Attribution " << split_string(attrs[i], "=")[0];
				}
			}
		}

		template<typename Dtype>
		int operation_lstm<Dtype>::init_weights(FILE* fp)
		{
			int quantize_tag;
			fread(&quantize_tag, 1, sizeof(int), fp);
			int mem = 0;
			if (quantize_tag == 0)
			{
				int num_directions = direction_ == 2 ? 2 : 1;
				this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(weight_data_size_, this->params_.device_, memory::NCHW, nullptr)));
				fread(this->weights_f32_[0]->mutable_cpu_data(), 1, weight_data_size_ * sizeof(float), fp);
				mem += weight_data_size_ * sizeof(float);
				this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(4 * num_output_ * num_directions, this->params_.device_, memory::NCHW, nullptr)));
				fread(this->weights_f32_[1]->mutable_cpu_data(), 1, 4 * num_output_ * num_directions * sizeof(float), fp);
				mem += 4 * num_output_ * num_directions * sizeof(float);
				this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(4 * num_output_ * num_output_ * num_directions, this->params_.device_, memory::NCHW, nullptr)));
				fread(this->weights_f32_[2]->mutable_cpu_data(), 1, 4 * num_output_ * num_output_ * num_directions * sizeof(float), fp);
				mem += 4 * num_output_ * num_output_ * num_directions * sizeof(float);
				return mem;
			}
			else
			{
				NOT_IMPLEMENTED;
				return 0;
			}
		}

		template<typename Dtype>
		int operation_lstm<Dtype>::init_weights()
		{
			std::default_random_engine e;
			std::normal_distribution<float> n(0, 0.3);
			std::uniform_int_distribution<int> u(-128, 127);
			int mem = 0;
			if (!this->params_.int8_quantization_)
			{
				int num_directions = direction_ == 2 ? 2 : 1;
				this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(weight_data_size_, this->params_.device_, memory::NCHW, nullptr)));
				for (size_t i = 0; i < weight_data_size_; i++)
				{
					this->weights_f32_[0]->mutable_cpu_data()[i] = n(e);
				}
				mem += weight_data_size_ * sizeof(float);
				this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(4 * num_output_ * num_directions, this->params_.device_, memory::NCHW, nullptr)));
				for (size_t i = 0; i < 4 * num_output_ * num_directions; i++)
				{
					this->weights_f32_[1]->mutable_cpu_data()[i] = n(e);
				}
				mem += 4 * num_output_ * num_directions * sizeof(float);
				this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(4 * num_output_ * num_output_ * num_directions, this->params_.device_, memory::NCHW, nullptr)));
				for (size_t i = 0; i < 4 * num_output_ * num_output_ * num_directions; i++)
				{
					this->weights_f32_[2]->mutable_cpu_data()[i] = n(e);
				}
				mem += 4 * num_output_ * num_output_ * num_directions * sizeof(float);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
			return mem;
		}

		template<typename Dtype>
		void operation_lstm<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			CHECK_EQ(tops.size(), 1);

			int num = bottoms[0]->num();
			int T = bottoms[0]->height();

			int num_directions = direction_ == 2 ? 2 : 1;
			if (bottoms[0]->order() == memory::NCHW)
			{
				tops[0].reset(new memory::tensor<float>(std::vector<int>{num, T, num_directions, num_output_}, this->params_.device_, memory::NCHW));
				if (direction_ == 0 || direction_ == 1)
				{
					lstm_cpu_f32(bottoms[0], tops[0], direction_);
				}
				else if (direction_ == 2)
				{
					auto top_forward = std::make_shared<memory::tensor<float>>(num, T, num_output_, this->params_.device_);
					auto top_reverse = std::make_shared<memory::tensor<float>>(num, T, num_output_, this->params_.device_);
					lstm_cpu_f32(bottoms[0], top_forward, 0);
					lstm_cpu_f32(bottoms[0], top_reverse, 1);

					const float* top_forward_data = top_forward->cpu_data();
					const float* top_reverse_data = top_reverse->cpu_data();
					for (int n = 0; n < num; n++)
					{
						for (int i = 0; i < T; i++)
						{
							const float* pf = top_forward_data + n * T * num_output_ + i * num_output_;
							const float* pr = top_reverse_data + n * T * num_output_ + i * num_output_;
							float* ptr = tops[0]->mutable_cpu_data() + n * T * num_output_ * num_directions + i * num_output_ * num_directions;

							std::copy(pf, pf + num_output_, ptr);
							std::copy(pr, pr + num_output_, ptr + num_output_);
						}
					}
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		template<typename Dtype>
		void operation_lstm<Dtype>::lstm_cpu_f32(const std::shared_ptr<memory::tensor<float>>& bottom, std::shared_ptr<memory::tensor<float>>& top, int reverse)
		{
			int num = bottom->num();
			int size = bottom->width();
			int T = bottom->height();

			int w_xc = this->weight_data_size_ / (direction_ == 2 ? 2 : 1) / num_output_ / 4;

			const float* weight_xc_data = this->weights_f32_[0]->cpu_data() + (direction_ == 2 ? reverse : 0) * w_xc * num_output_ * 4;

			const float* bias_c_data = this->weights_f32_[1]->cpu_data() + (direction_ == 2 ? reverse : 0) * 4 * num_output_;
			const float* bias_c_I = bias_c_data + 0 * num_output_;
			const float* bias_c_F = bias_c_data + 1 * num_output_;
			const float* bias_c_O = bias_c_data + 2 * num_output_;
			const float* bias_c_G = bias_c_data + 3 * num_output_;

			const float* weight_hc_data = this->weights_f32_[2]->cpu_data() + (direction_ == 2 ? reverse : 0) * num_output_ * num_output_ * 4;

			float* hidden_data = hidden_->mutable_cpu_data();
			float* cell_data = cell_->mutable_cpu_data();
			float* gates = gates_->mutable_cpu_data();

			for (int n = 0; n < num; n++)
			{
				std::memset(hidden_data, 0, num_output_ * sizeof(float));
				std::memset(cell_data, 0, num_output_ * sizeof(float));
				const float* bottom_data = bottom->cpu_data() + n * size * T;
				float* top_data = top->mutable_cpu_data() + n * T * num_output_;
				// unroll
				for (int t = 0; t < T; t++)
				{
					// clip hidden by continuation indicator
					// h_cont_{t-1} = cont_t * h_{t-1}
					// h_cont_{t-1} = h_{t-1} if cont_t == 1
					//                0       otherwise
					// calculate hidden
					// gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c

					int ti = reverse ? T - 1 - t : t;

					const float* x = bottom_data + ti * size;
					for (int q = 0; q < num_output_; q++)
					{

						float* gates_data = gates + q * 4;

						// gate I F O G
						const float* weight_xc_I = weight_xc_data + (num_output_ * 0 + q) * w_xc;
						const float* weight_xc_F = weight_xc_data + (num_output_ * 1 + q) * w_xc;
						const float* weight_xc_O = weight_xc_data + (num_output_ * 2 + q) * w_xc;
						const float* weight_xc_G = weight_xc_data + (num_output_ * 3 + q) * w_xc;

						const float* weight_hc_I = weight_hc_data + (num_output_ * 0 + q) * num_output_;
						const float* weight_hc_F = weight_hc_data + (num_output_ * 1 + q) * num_output_;
						const float* weight_hc_O = weight_hc_data + (num_output_ * 2 + q) * num_output_;
						const float* weight_hc_G = weight_hc_data + (num_output_ * 3 + q) * num_output_;

						float I = bias_c_I[q];
						float F = bias_c_F[q];
						float O = bias_c_O[q];
						float G = bias_c_G[q];

						for (int i = 0; i < size; i++)
						{
							float xi = x[i];

							I += weight_xc_I[i] * xi;
							F += weight_xc_F[i] * xi;
							O += weight_xc_O[i] * xi;
							G += weight_xc_G[i] * xi;
						}

						for (int i = 0; i < num_output_; i++)
						{
							float h_cont = hidden_data[i];

							I += weight_hc_I[i] * h_cont;
							F += weight_hc_F[i] * h_cont;
							O += weight_hc_O[i] * h_cont;
							G += weight_hc_G[i] * h_cont;
						}

						gates_data[0] = I;
						gates_data[1] = F;
						gates_data[2] = O;
						gates_data[3] = G;
					}

					// lstm unit
					// sigmoid(I)
					// sigmoid(F)
					// sigmoid(O)
					// tanh(G)
					// c_t := f_t .* c_{t-1} + i_t .* g_t
					// h_t := o_t .* tanh[c_t]
					float* output_data = top_data + ti * num_output_;
					for (int q = 0; q < num_output_; q++)
					{
						const float* gates_data = gates + q * 4;

						float I = gates_data[0];
						float F = gates_data[1];
						float O = gates_data[2];
						float G = gates_data[3];

						I = 1.f / (1.f + exp(-I));
						F = 1.f / (1.f + exp(-F));
						O = 1.f / (1.f + exp(-O));
						G = std::tanh(G);

						float cell2 = F * cell_data[q] + I * G;
						float H = O * std::tanh(cell2);
						cell_data[q] = cell2;
						hidden_data[q] = H;
						output_data[q] = H;
					}
				}
			}
		}

		template<typename Dtype>
		void operation_lstm<Dtype>::forward_gpu_f32(
#ifdef USE_CUDA
			cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			NOT_IMPLEMENTED;
		}

//#ifndef USE_CUDA
//		STUB_GPU(operation_lstm);
//#endif
		INSTANCE_CLASS(operation_lstm);
		REGISTE(operation_lstm);
	}
}