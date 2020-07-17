#include "../../include/Excalibur/operation_eltwise.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
	namespace excalibur
	{
#ifdef USE_CUDA
		__global__ void max_kernel(const int nthreads, float* bottom_data_a, float* bottom_data_b, float* top_data) 
		{
			CUDA_KERNEL_LOOP(index, nthreads) 
			{
				top_data[index] = max(bottom_data_a[index], bottom_data_b[index]);
			}
		}

		__global__ void mul_kernel(const int nthreads, float* bottom_data_a, float* bottom_data_b, float* top_data)
		{
			CUDA_KERNEL_LOOP(index, nthreads)
			{
				top_data[index] = bottom_data_a[index] * bottom_data_b[index];
		}
	}

		template<typename Dtype>
		void operation_eltwise<Dtype>::forward_gpu_f32(
			cublasHandle_t &cublas_handle,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_GE(bottoms.size(), 2);
			for (int i = 1; i < bottoms.size(); ++i)
			{
				CHECK(bottoms[i]->data_shape() == bottoms[0]->data_shape());
			}
			CHECK_EQ(tops.size(), 1);
			tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
			float* bottom_data_a = nullptr;
			float* bottom_data_b = nullptr;
			const int count = tops[0]->count();
			float* top_data = tops[0]->mutable_gpu_data();
			switch (type_)
			{
			case SUM:
				CUDA_CHECK(cudaMemset(top_data, 0, count * sizeof(float)));
				for (int i = 0; i < bottoms.size(); ++i)
				{
					CUBLAS_CHECK(cublasSaxpy(cublas_handle, count, &coeffs_[i], bottoms[i]->gpu_data(), 1, top_data, 1));
				}
				break;
			case MAX:
				bottom_data_a = bottoms[0]->mutable_gpu_data();
				bottom_data_b = bottoms[1]->mutable_gpu_data();
				max_kernel << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (
					count, bottom_data_a, bottom_data_b, top_data);
				for (int blob_idx = 2; blob_idx < bottoms.size(); ++blob_idx)
				{
					bottom_data_b = bottoms[blob_idx]->mutable_gpu_data();
					max_kernel << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (
						count, top_data, bottom_data_b, top_data);
				}
				break;
			case PROD:
				bottom_data_a = bottoms[0]->mutable_gpu_data();
				bottom_data_b = bottoms[1]->mutable_gpu_data();
				mul_kernel << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (
					count, bottom_data_a, bottom_data_b, top_data);
				for (int blob_idx = 2; blob_idx < bottoms.size(); ++blob_idx)
				{
					bottom_data_b = bottoms[blob_idx]->mutable_gpu_data();
					mul_kernel << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (
						count, top_data, bottom_data_b, top_data);
				}
				break;
			default:
				LOG(FATAL) << "Unknown elementwise operation.";
				break;
			}
		}

#ifdef USE_CUDNN
		INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_eltwise);
#else
		INSTANTIATE_OPERATION_CUDA_FWDF32(operation_eltwise);
#endif

#endif // USE_CUDA
	}
}