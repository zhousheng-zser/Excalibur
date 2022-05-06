#include "../../../include/Excalibur/operation_elu.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"

#ifdef USE_CUDA
#include <cuda_fp16.hpp>
#endif

namespace glasssix
{
	namespace excalibur
	{
#ifdef USE_CUDA
		__global__ void elu_kernel(const int n, const float* in, float* out, float alpha)
		{
			CUDA_KERNEL_LOOP(index, n)
			{
				out[index] = in[index] < 0 ? (exp(in[index])-1)*alpha : in[index];
			}
		}

		template<typename Dtype>
		void operation_elu<Dtype>::forward_gpu_f32(
			cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), tops.size());
			for (size_t i = 0; i < bottoms.size(); i++)
			{
				tops[i].reset(new memory::tensor<float>(bottoms[i]->data_shape(), this->params_.device_, bottoms[i]->order(), bottoms[i]->allocator()));
				float* top_data = tops[i]->mutable_gpu_data();
				const float* bottom_data = bottoms[i]->gpu_data();
				const int count = bottoms[i]->count();
				elu_kernel << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (
					count, bottom_data, top_data, this->alpha_);
			}
			CUDA_POST_KERNEL_CHECK;
		}

		__global__ void elu_kernel_f16(const int n, const unsigned short* in, unsigned short* out, float alpha)
		{
			__half alpha_h = __float2half(alpha);
			__half zero = __float2half(0.0f);
			__half one = __float2half(1.0f);
			CUDA_KERNEL_LOOP(index, n)
			{
				__half value = __ushort_as_half(in[index]);
				out[index] = __half_as_ushort(__hlt(value, zero) ? __hmul(__hsub(hexp(value), one), alpha_h) : value);
			}
		}

		template<typename Dtype>
		void operation_elu<Dtype>::forward_gpu_f16(
			cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
			const std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& tops)
		{
			CHECK_EQ(bottoms.size(), tops.size());
			for (size_t i = 0; i < bottoms.size(); i++)
			{
				tops[i].reset(new memory::tensor<unsigned short>(bottoms[i]->data_shape(), this->params_.device_, bottoms[i]->order(), bottoms[i]->allocator()));
				unsigned short* top_data = tops[i]->mutable_gpu_data();
				const unsigned short* bottom_data = bottoms[i]->gpu_data();
				const int count = bottoms[i]->count();
				elu_kernel_f16 << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (
					count, bottom_data, top_data, this->alpha_);
			}
			CUDA_POST_KERNEL_CHECK;
		}

#ifdef USE_CUDNN
		INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_elu);
		INSTANTIATE_OPERATION_CUDNN_FWDF16(operation_elu);
#else
		INSTANTIATE_OPERATION_CUDA_FWDF32(operation_elu);
		INSTANTIATE_OPERATION_CUDA_FWDF16(operation_elu);
#endif

#endif // USE_CUDA

	}
}