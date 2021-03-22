#include "../../../include/Excalibur/operation_prelu.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"
#include "../../../include/Excalibur/math_functions.hpp"

namespace glasssix
{
	namespace excalibur
	{
#ifdef USE_CUDA

		__global__ void PReLUForward(const int n, const int channels, const int dim,
			const float* in, float* out, const float* slope_data,
			const int div_factor, memory::orderType order) {

			if (order == memory::NCHW)
			{
				CUDA_KERNEL_LOOP(index, n) {
					int c = (index / dim) % channels / div_factor;
					out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
				}
			}
			else if (order == memory::NHWC)
			{
				CUDA_KERNEL_LOOP(index, n) {
					int c = index % channels / div_factor;
					out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
				}
			}
			else
			{
				return;
			}
		}

		template<typename Dtype>
		void operation_prelu<Dtype>::forward_gpu_f32(
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
				const float* bottom_data = bottoms[i]->gpu_data();
				float* top_data = tops[i]->mutable_gpu_data();
				const int count = bottoms[i]->count();
				memory::orderType order = bottoms[i]->order();
				int dim;
				if (bottoms[i]->data_shape().size() <= 2)
				{
					dim = 1;
				}
				else
				{
					dim = bottoms[i]->height() * bottoms[i]->width();
				}
				const int channels = bottoms[i]->channels();
				const float* slope_data = this->weights_f32_[0]->gpu_data();
				const int div_factor = false ? channels : 1;

				// NOLINT_NEXT_LINE(whitespace/operators)
				PReLUForward << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (
					count, channels, dim, bottom_data, top_data, slope_data, div_factor, order);
			}
			CUDA_POST_KERNEL_CHECK;
		}

#ifdef USE_CUDNN
		INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_prelu);
#else
		INSTANTIATE_OPERATION_CUDA_FWDF32(operation_prelu);
#endif

#endif
	}
}