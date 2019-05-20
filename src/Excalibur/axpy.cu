#ifdef USE_CUDA
#include "math_functions.hpp"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "axpy.hpp"

namespace glasssix
{
	namespace excalibur
	{
		__global__ void kernel_axpy(const float *bottom_data, const float *scales_data, int channels, int spatial_dim, float *top_data) 
		{
			int totalID = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int numID = totalID / (channels * spatial_dim);
			int remainID = totalID % (channels * spatial_dim);
			int channelID = remainID % channels;
			top_data[totalID] += bottom_data[totalID] * scales_data[numID * channels + channelID];
		}

		void axpy::Forward_gpu_native(cublasHandle_t cublas_handle_, const std::vector<std::shared_ptr<tensor<float>>> bottom, std::shared_ptr<tensor<float>>& top)
		{
			CHECK_EQ(bottom.size(), 2);
			scales_ = bottom[0];
			input_ = bottom[1];
			order_ = input_->order();
			CHECK_EQ((scales_->data_shape()).size(), 2);
			CHECK_EQ(scales_->data_shape()[0], input_->num());
			CHECK_EQ(scales_->data_shape()[1], input_->channels());

			if (top == nullptr)
			{
				top.reset(new tensor<float>(input_->data_shape(), device_, order_));
			}
			else
			{
				CHECK_EQ(input_->data_shape()[0], top->data_shape()[0]);
				CHECK_EQ(input_->data_shape()[1], top->data_shape()[1]);
				CHECK_EQ(input_->data_shape()[2], top->data_shape()[2]);
				CHECK_EQ(input_->data_shape()[3], top->data_shape()[3]);

				if (order_ != top->order())
				{
					LOG(ERROR) << "order should be consistent!!!";
					return;
				}

				if (device_ != top->device())
				{
					LOG(ERROR) << "device should be consistent!!!";
					return;
				}
			}

			int num = input_->data_shape()[0];
			const float *scales_data = scales_->gpu_data();
			const float *bottom_data = input_->gpu_data();
			float *top_data = top->mutable_gpu_data();

			if (order_ == NCHW)
			{
				int channels = input_->data_shape()[1];
				int spatial_dim = input_->count(2, 4);

				for (int n = 0; n < num; ++n)
				{
					for (int ch = 0; ch < channels; ch++)
					{
						int scale_offset = n * channels + ch;
						int data_offset = scale_offset * spatial_dim;
						math_functions::gpu_saxpy(cublas_handle_, spatial_dim, scales_data[scale_offset], bottom_data + data_offset, top_data + data_offset);
					}
				}
			}
			else if (order_ == NHWC)
			{
				int channels = input_->data_shape()[3];
				int spatial_dim = input_->count(1, 3);

				const dim3 block_size(channels, 1, 1);
				const dim3 grid_size(num, spatial_dim, 1);

				kernel_axpy << <grid_size, block_size >> > (bottom_data, scales_data, channels, spatial_dim, top_data);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}
	}
}


#endif