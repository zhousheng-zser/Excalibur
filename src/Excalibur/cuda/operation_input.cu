#include "../../../include/Excalibur/operation_input.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"

#ifdef USE_CUDA
#include <cuda_fp16.hpp>
#endif

namespace glasssix
{
    namespace excalibur
    {
#ifdef USE_CUDA

        __global__ void input_forward(int n, const float *bottom_data, int channels, int height, int width, memory::orderType order, float *top_data, const float *means_var)
        {
            if (order == memory::NCHW)
            {
                CUDA_KERNEL_LOOP(index, n)
                {
                    int c = (index / height / width) % channels;
                    top_data[index] = (bottom_data[index] - means_var[c]) * means_var[channels];
                }
            }
            else if (order == memory::NHWC)
            {
                CUDA_KERNEL_LOOP(index, n)
                {
                    int c = index % channels;
                    top_data[index] = (bottom_data[index] - means_var[c]) * means_var[channels];
                }
            }
            else
            {
                return;
            }
        }

        template <typename Dtype>
        void operation_input<Dtype>::forward_gpu_f32(
            cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
            cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
            const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms,
            std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            memory::tensor<float> means_var(means_.size() + 1, this->params_.device_, memory::NCHW, bottoms[0]->allocator());
            std::copy(means_.data(), means_.data() + means_.size(), means_var.mutable_cpu_data());
            *(means_var.mutable_cpu_data() + means_.size()) = var_;

            for (size_t i = 0; i < bottoms.size(); i++)
            {
            	tops[i].reset(new memory::tensor<float>(bottoms[i]->data_shape(), this->params_.device_, bottoms[i]->order(), bottoms[i]->allocator()));
            	int count = bottoms[i]->count();
            	int channels = bottoms[i]->channels();
            	int height = bottoms[i]->height();
            	int width = bottoms[i]->width();
            	int offset = height * width;
            	float* top_data = tops[i]->mutable_gpu_data();
            	const float* bottom_data = bottoms[i]->gpu_data();

            	if(channels != 3 && channels != 1)
            	{
            		LOG(FATAL) << "Un-supprted channel num: " << channels;
            	}

            	input_forward << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (
            		count, bottom_data, channels, height, width, bottoms[i]->order(), top_data, means_var.gpu_data());
            }
            CUDA_POST_KERNEL_CHECK;
            // tops = bottoms;
        }

		__global__ void input_forward_f16(int n, const unsigned short* bottom_data, int channels, int height, int width, memory::orderType order, unsigned short* top_data, const unsigned short* means_var)
		{
			if (order == memory::NCHW)
			{
				CUDA_KERNEL_LOOP(index, n) 
				{
					int c = (index / height / width) % channels;
					top_data[index] = __half_as_ushort(__hmul(__hsub(__ushort_as_half(bottom_data[index]), __ushort_as_half(means_var[c])), __ushort_as_half(means_var[channels])));
				}
			}
			else if (order == memory::NHWC)
			{
				CUDA_KERNEL_LOOP(index, n) {
					int c = index % channels;
					top_data[index] = __half_as_ushort(__hmul(__hsub(__ushort_as_half(bottom_data[index]), __ushort_as_half(means_var[c])), __ushort_as_half(means_var[channels])));
				}
			}
			else
			{
				return;
			}
		}

		template<typename Dtype>
		void operation_input<Dtype>::forward_gpu_f16(
			cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
			const std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& tops)
		{
			memory::tensor<unsigned short> means_var(means_.size() + 1, this->params_.device_, memory::NCHW, bottoms[0]->allocator());
			unsigned short* means_var_data = means_var.mutable_cpu_data();
			float2half(means_.data(), means_var_data, means_.size());
			*(means_var_data + means_.size()) = float32_to_float16(var_);

			for (size_t i = 0; i < bottoms.size(); i++)
			{
				tops[i].reset(new memory::tensor<unsigned short>(bottoms[i]->data_shape(), this->params_.device_, bottoms[i]->order(), bottoms[i]->allocator()));
				int count = bottoms[i]->count();
				int channels = bottoms[i]->channels();
				int height = bottoms[i]->height();
				int width = bottoms[i]->width();
				int offset = height * width;
				unsigned short* top_data = tops[i]->mutable_gpu_data();
				const unsigned short* bottom_data = bottoms[i]->gpu_data();

				if (channels != 3 && channels != 1)
				{
					LOG(FATAL) << "Un-supprted channel num: " << channels;
				}

				input_forward_f16 << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (
					count, bottom_data, channels, height, width, bottoms[i]->order(), top_data, means_var.gpu_data());
			}
			CUDA_POST_KERNEL_CHECK;
		}

#ifdef USE_CUDNN
		INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_input);
		INSTANTIATE_OPERATION_CUDNN_FWDF16(operation_input);
#else
		INSTANTIATE_OPERATION_CUDA_FWDF32(operation_input);
		INSTANTIATE_OPERATION_CUDA_FWDF16(operation_input);
#endif

#endif
    }
}