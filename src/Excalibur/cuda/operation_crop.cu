#include "../../../include/Excalibur/operation_crop.hpp"
#include "../../../include/Excalibur/math_functions.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
	namespace excalibur
	{
#ifdef USE_CUDA

		__global__ void crop_forward_nchw(int n, const float* bottom_data, int bottom_offset_n, int bottom_offset_c, int bottom_offset_h,
			float* top_data, int top_offset_n, int top_offset_c, int top_offset_h, int coffset, int hoffset, int woffset)
		{
			CUDA_KERNEL_LOOP(index, n) {
				int a = index % top_offset_n;
				int b = a / top_offset_c;
				int c = a % top_offset_c;
				int d = c / top_offset_h;
				int e = c % top_offset_h;

				int f = (index / top_offset_n) * bottom_offset_n 
					+ (b + coffset) * bottom_offset_c
					+ (d + hoffset) * bottom_offset_h
					+ (e + woffset);

				top_data[index] = bottom_data[f];
			}
		}

		__global__ void crop_forward_nhwc(int n, const float* bottom_data, int bottom_offset_n, int bottom_channels, int bottom_offset_h,
			float* top_data, int top_offset_n, int top_offset_h, int top_channels, int coffset, int hoffset, int woffset)
		{
			CUDA_KERNEL_LOOP(index, n) {
				int a = index % top_offset_n;
				int b = a / top_offset_h;
				int c = a % top_offset_h;
				int d = c / top_channels;
				int e = c % top_channels;

				int f = (index / top_offset_n) * bottom_offset_n
					+ (b + hoffset) * bottom_offset_h
					+ (d + woffset) * bottom_channels;
					+ (e + coffset);

				top_data[index] = bottom_data[f];
			}
		}

		template<typename Dtype>
		void operation_crop<Dtype>::forward_gpu_f32(
			cublasHandle_t& cublas_handle_,
		#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
		#endif //!USE_CUDNN
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 2);
			CHECK_EQ(tops.size(), 1);
			CHECK_EQ(bottoms[0]->order(), bottoms[1]->order());
			CHECK_EQ(bottoms[0]->num(), bottoms[1]->num());
			CHECK_LE(bottoms[1]->channels() + coffset_, bottoms[0]->channels());
			CHECK_LE(bottoms[1]->height() + hoffset_, bottoms[0]->height());
			CHECK_LE(bottoms[1]->width() + woffset_, bottoms[0]->width());
			tops[0].reset(new memory::tensor<float>(bottoms[1]->data_shape(), this->params_.device_, bottoms[1]->order(), bottoms[1]->allocator()));
			auto top_data = tops[0]->mutable_gpu_data();
			auto bottom_data = bottoms[0]->gpu_data();
			int count = bottoms[1]->count();
			if (bottoms[1]->order() == memory::NCHW)
			{
				int bottom_offset_n = bottoms[0]->count(1, 4);
				int top_offset_n = tops[0]->count(1, 4);
				int bottom_offset_c = bottoms[0]->count(2, 4);
				int top_offset_c = tops[0]->count(2, 4);
				int bottom_offset_h = bottoms[0]->count(3, 4);
				int top_offset_h = tops[0]->count(3, 4);

				crop_forward_nchw << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (
					count, bottom_data, bottom_offset_n, bottom_offset_c, bottom_offset_h, 
					top_data, top_offset_n, top_offset_c, top_offset_h, 
					coffset_, hoffset_, woffset_);
			}
			else
			{
				int bottom_offset_n = bottoms[0]->count(1, 4);
				int top_offset_n = tops[0]->count(1, 4);
				int bottom_offset_h = bottoms[0]->count(2, 4);
				int top_offset_h = tops[0]->count(2, 4);

				crop_forward_nhwc << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (
					count, bottom_data, bottom_offset_n, bottom_offset_h, bottoms[0]->channels(),
					top_data, top_offset_n, top_offset_h, tops[0]->channels(),
					coffset_, hoffset_, woffset_);
			}
		}

#ifdef USE_CUDNN
INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_crop);
#else
INSTANTIATE_OPERATION_CUDA_FWDF32(operation_crop);
#endif

#endif
	}
}