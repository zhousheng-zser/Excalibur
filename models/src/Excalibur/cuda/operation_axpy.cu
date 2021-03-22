#include "../../../include/Excalibur/operation_axpy.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"
#include "../../../include/Excalibur/math_functions.hpp"

namespace glasssix
{
	namespace excalibur
	{
#ifdef USE_CUDA

		__global__ void kernel_axpy(const float* bottom_data, const float* scales_data, int channels, int spatial_dim, float* top_data)
		{
			int totalID = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int numID = totalID / (channels * spatial_dim);
			int remainID = totalID % (channels * spatial_dim);
			int channelID = remainID % channels;
			top_data[totalID] += bottom_data[totalID] * scales_data[numID * channels + channelID];
		}

		template<typename Dtype>
		void operation_axpy<Dtype>::forward_gpu_f32(
			cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 2);
			CHECK_EQ(tops.size(), 1);
			CHECK_EQ(bottoms[0]->num(), bottoms[1]->num());
			CHECK_EQ(bottoms[0]->channels(), bottoms[1]->channels());
			CHECK_EQ(bottoms[1]->width(), 1);
			CHECK_EQ(bottoms[1]->height(), 1);

			tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), this->params_.device_, bottoms[0]->order(), bottoms[0]->allocator()));

			int num = bottoms[0]->num();
			const float *bottom_data = bottoms[0]->gpu_data();
			const float *scales_data = bottoms[1]->gpu_data();
			float *top_data = tops[0]->mutable_gpu_data();

			memory::orderType order = bottoms[0]->order();
			if (order == memory::NCHW)
			{
				int channels = bottoms[0]->channels();
				int spatial_dim = bottoms[0]->count(2, 4);

				for (int n = 0; n < num; ++n)
				{
					for (int ch = 0; ch < channels; ch++)
					{
						int scale_offset = n * channels + ch;
						int data_offset = scale_offset * spatial_dim;
						math_functions::gpu_saxpy(cublas_handle_, spatial_dim, scales_data[scale_offset], bottom_data + data_offset, top_data + data_offset);
					}
				}
				CUDA_POST_KERNEL_CHECK;
			}
			else if (order == memory::NHWC)
			{
				int channels = bottoms[0]->channels();
				int spatial_dim = bottoms[0]->count(1, 3);

				const dim3 block_size(channels, 1, 1);
				const dim3 grid_size(num, spatial_dim, 1);

				kernel_axpy << <grid_size, block_size >> > (bottom_data, scales_data, channels, spatial_dim, top_data);
				CUDA_POST_KERNEL_CHECK;
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

#ifdef USE_CUDNN
		INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_axpy);
#else
		INSTANTIATE_OPERATION_CUDA_FWDF32(operation_axpy);
#endif

#endif
	}
}