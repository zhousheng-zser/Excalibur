#include "../../../include/Excalibur/operation_shufflechannel.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"

#include "../../../include/Excalibur/math_functions.hpp"

namespace glasssix
{
	namespace excalibur
	{
#ifdef USE_CUDA

		template<typename Dtype>
		void operation_shufflechannel<Dtype>::forward_gpu_f32(
			cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			CHECK_EQ(tops.size(), 1);

			int num = bottoms[0]->num();
			int w = bottoms[0]->width();
			int h = bottoms[0]->height();
			int channels = bottoms[0]->channels();

			CHECK_EQ(channels % group_, 0);

			int _group = reverse_ ? channels / group_ : group_;
			int channels_per_group = channels / _group;

			if (bottoms[0]->order() == memory::NHWC)
				NOT_IMPLEMENTED;

			tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));

			int c_step = w * h;
			for (size_t n = 0; n < num; n++)
			{
				const float* bottom_data = bottoms[0]->gpu_data();
				float* top_data = tops[0]->mutable_gpu_data();
				for (int i = 0; i < _group; i++)
				{
					for (int j = 0; j < channels_per_group; j++)
					{
						int src_coffset = channels_per_group * i + j;
						int dst_coffset = _group * j + i;
						cudaMemcpy(top_data + dst_coffset * c_step, bottom_data + src_coffset * c_step, sizeof(float) * c_step, cudaMemcpyDeviceToDevice);
					}
				}
			}
		}

		
#ifdef USE_CUDNN
		INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_shufflechannel);
#else
		INSTANTIATE_OPERATION_CUDA_FWDF32(operation_shufflechannel);
#endif

#endif // USE_CUDA

	}
}