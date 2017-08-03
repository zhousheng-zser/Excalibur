#include "ReLU_Layer.hpp"

namespace Excalibur
{
#ifdef USE_CUDA
	template <typename Dtype>
	__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out) 
	{
		CUDA_KERNEL_LOOP(index, n) 
		{
			out[index] = in[index] > 0 ? in[index] : 0;
		}
	}

	template <typename Dtype>
	void ReLU_Layer<Dtype>::Forward_gpu(const std::vector<Pandora_Blob<Dtype>*>& bottom,
		const std::vector<Pandora_Blob<Dtype>*>& top)
	{
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		const int count = bottom[0]->count();
		ReLUForward<Dtype><<< EXCALIBUR_GET_BLOCKS(count), EXCALIBUR_CUDA_NUM_THREADS >>>(count, bottom_data, top_data);
		CUDA_POST_KERNEL_CHECK;
	}

	template class ReLU_Layer<float>;
	template class ReLU_Layer<double>;
#endif
}