#include "ReLU_Layer.hpp"


namespace Excalibur
{
	template <typename Dtype>
	ReLU_Layer<Dtype>::ReLU_Layer()
	{
	}

	template <typename Dtype>
	ReLU_Layer<Dtype>::~ReLU_Layer()
	{
	}

	template <typename Dtype>
	std::string ReLU_Layer<Dtype>::test()
	{
		return this->type;
	}

	template <typename Dtype>
	void ReLU_Layer<Dtype>::Reshape(const std::vector<Pandora_Blob<Dtype>*>& bottom,
		const std::vector<Pandora_Blob<Dtype>*>& top)
	{
		
	}

	template <typename Dtype>
	void ReLU_Layer<Dtype>::Forward_cpu(const std::vector<Pandora_Blob<Dtype>*>& bottom,
		const std::vector<Pandora_Blob<Dtype>*>& top)
	{
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		const int count = bottom[0]->count();
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < count; i++)
		{
			top_data[i] = std::max(bottom_data[i], Dtype(0));
		}
		this->exmath_->excalibur_cpu_sub(count, bottom_data, top_data, top_data);
	}

	template class ReLU_Layer<float>;
	template class ReLU_Layer<double>;
}