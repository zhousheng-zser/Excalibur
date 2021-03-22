#include "../../../include/Excalibur/operation_relu.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
	namespace excalibur
	{
#ifdef USE_CUDA
		__global__ void relu_kernel(const int n, const float* in, float* out)
		{
			CUDA_KERNEL_LOOP(index, n)
			{
				out[index] = in[index] > 0 ? in[index] : 0.0f;
			}
		}

		template<typename Dtype>
		void operation_relu<Dtype>::forward_gpu_f32(
			cublasHandle_t &cublas_handle_,
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
				relu_kernel << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (
					count, bottom_data, top_data);
			}
		}

#ifdef USE_CUDNN
		INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_relu);
#else
		INSTANTIATE_OPERATION_CUDA_FWDF32(operation_relu);
#endif

#endif // USE_CUDA

	}
}