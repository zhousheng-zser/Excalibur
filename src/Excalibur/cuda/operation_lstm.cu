#include "../../../include/Excalibur/operation_lstm.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"

#include <cfloat>

namespace glasssix
{
    namespace excalibur
    {
#ifdef USE_CUDA
        __global__ void lstm_get_gates(
            const int N,
            const float *bottom_data,
            int ti,
            int size,
            const float *weight_xc_data,
            int num_output,
            int w_xc,
            const float *weight_hc_data,
            const float *bias_c_data,
            float *hidden_data,
            float *gates)
        {
            CUDA_KERNEL_LOOP(index, N)
            {
                const float *x = bottom_data + ti * size;
                const float *weight_xc_ptr = weight_xc_data + index * w_xc;
                const float *weight_hc_ptr = weight_hc_data + index * num_output;

                float val = bias_c_data[index];
                for (int i = 0; i < size; i++)
                {
                    float xi = x[i];
                    val += weight_xc_ptr[i] * xi;
                }

                for (int i = 0; i < num_output; i++)
                {
                    float h_cont = hidden_data[i];
                    val += weight_hc_ptr[i] * h_cont;
                }
                int k = index % num_output * 4 + index / num_output;
                gates[k] = val;
            }
        }

        __global__ void lstm_forward(
            int num_output,
            float *gates,
            float *cell_data,
            float *hidden_data,
            float *output_data)
        {
            CUDA_KERNEL_LOOP(index, num_output)
            {
                const float *gates_data = gates + index * 4;
                float I = gates_data[0];
                float F = gates_data[1];
                float O = gates_data[2];
                float G = gates_data[3];

                I = 1.f / (1.f + exp(-I));
                F = 1.f / (1.f + exp(-F));
                O = 1.f / (1.f + exp(-O));
                G = tanh(G);

                float cell2 = F * cell_data[index] + I * G;
                float H = O * tanh(cell2);
                cell_data[index] = cell2;
                hidden_data[index] = H;
                output_data[index] = H;
            }
        }

        template <class Dtype>
        void operation_lstm<Dtype>::forward_gpu_f32(
#ifdef USE_CUDA
            cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
            cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
            const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms,
            std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);
            int num = bottoms[0]->num();

            int T = bottoms[0]->height();
            int num_directions = direction_ == 2 ? 2 : 1;

            if (bottoms[0]->order() == memory::NCHW)
            {
                tops[0].reset(new memory::tensor<float>(std::vector<int>{num, 1, T, num_directions * num_output_}, this->params_.device_, memory::NCHW));
                if (direction_ == 0 || direction_ == 1)
                {
                    lstm_gpu_f32(bottoms[0], tops[0], direction_);
                }
                else if (direction_ == 2)
                {
                    auto top_forward = std::make_shared<memory::tensor<float>>(num, T, num_output_, this->params_.device_);
                    auto top_reverse = std::make_shared<memory::tensor<float>>(num, T, num_output_, this->params_.device_);
                    lstm_gpu_f32(bottoms[0], top_forward, 0);
                    lstm_gpu_f32(bottoms[0], top_reverse, 1);

                    const float *top_forward_data = top_forward->gpu_data();
                    const float *top_reverse_data = top_reverse->gpu_data();
                    for (int n = 0; n < num; n++)
                    {
                        for (int i = 0; i < T; i++)
                        {
                            const float *pf = top_forward_data + n * T * num_output_ + i * num_output_;
                            const float *pr = top_reverse_data + n * T * num_output_ + i * num_output_;
                            float *ptr = tops[0]->mutable_gpu_data() + n * T * num_output_ * num_directions + i * num_output_ * num_directions;

                            cudaMemcpy(ptr, pf, num_output_ * sizeof(float), cudaMemcpyDeviceToDevice);
                            cudaMemcpy(ptr + num_output_, pr, num_output_ * sizeof(float), cudaMemcpyDeviceToDevice);
                        }
                    }
                }
            }
            else
            {
                NOT_IMPLEMENTED;
            }
        }

        template <typename Dtype>
        void operation_lstm<Dtype>::lstm_gpu_f32(const std::shared_ptr<memory::tensor<float>> &bottom, std::shared_ptr<memory::tensor<float>> &top, int reverse)
        {
            int num = bottom->num();
            int size = bottom->width();
            int T = bottom->height();

            int w_xc = this->weight_data_size_ / (direction_ == 2 ? 2 : 1) / num_output_ / 4;
            const float *weight_xc_data = this->weights_f32_[0]->gpu_data() + reverse * w_xc * num_output_ * 4;
            const float *bias_c_data = this->weights_f32_[1]->gpu_data() + reverse * 4 * num_output_;
            const float *weight_hc_data = this->weights_f32_[2]->gpu_data() + reverse * num_output_ * num_output_ * 4;

            float *hidden_data = hidden_->mutable_gpu_data();
            float *cell_data = cell_->mutable_gpu_data();
            float *gates = gates_->mutable_gpu_data();
            const int gates_count = gates_->count();

            for (int n = 0; n < num; n++)
            {
                cudaMemset(hidden_->mutable_gpu_data(), 0, num_output_ * sizeof(float));
                cudaMemset(cell_->mutable_gpu_data(), 0, num_output_ * sizeof(float));

                const float *bottom_data = bottom->gpu_data() + n * size * T;
                float *top_data = top->mutable_gpu_data() + n * T * num_output_;

                for (int t = 0; t < T; t++)
                {
                    int ti = reverse ? T - 1 - t : t;
                    lstm_get_gates<<<CUDA_GET_BLOCKS(gates_count), CUDA_NUM_THREADS>>>(
                        gates_count,
                        bottom_data,
                        ti,
                        size,
                        weight_xc_data,
                        num_output_,
                        w_xc,
                        weight_hc_data,
                        bias_c_data,
                        hidden_data,
                        gates);

                    float *output_data = top_data + ti * num_output_;
                    lstm_forward<<<CUDA_GET_BLOCKS(num_output_), CUDA_NUM_THREADS>>>(
                        num_output_,
                        gates,
                        cell_data,
                        hidden_data,
                        output_data);
                }
            }
        }

#ifdef USE_CUDNN
        INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_lstm);
#else
        INSTANTIATE_OPERATION_CUDA_FWDF32(operation_lstm);
#endif

#endif
    }
}