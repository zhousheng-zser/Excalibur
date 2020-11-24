#include "../../include/Excalibur/operation_relu6.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_relu6<Dtype>::operation_relu6(const operation_param& param) : operation<Dtype>(param)
		{
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					//do nothing
				}
				else if (split_string(attrs[i], "=")[0] == "1")
				{
					//do nothing
				}
				else if (split_string(attrs[i], "=")[0] == "-23330")
				{
					//do nothing
				}
				else
				{
					LOG(FATAL) << "Un-supported ReLU6 Attribution " << split_string(attrs[i], "=")[0];
				}
			}
			this->params_.inplace_ = true;
		}

		template<typename Dtype>
		void operation_relu6<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), tops.size());
			for (size_t i = 0; i < bottoms.size(); i++)
			{
				tops[i].reset(new memory::tensor<float>(bottoms[i]->data_shape(), bottoms[i]->device(), bottoms[i]->order(), bottoms[i]->allocator()));
				float* top_data = tops[i]->mutable_cpu_data();
				const float* bottom_data = bottoms[i]->cpu_data();
				const int count = bottoms[i]->count();
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (int j = 0; j < bottoms[i]->count(); j++)
				{
					top_data[j] = (bottom_data[j] >= 0.0f && bottom_data[j] <= 6.0f) ? bottom_data[j] : 0.0f;
				}
			}
		}

#ifndef USE_CUDA
		STUB_GPU(operation_relu6);
#endif

		INSTANCE_CLASS(operation_relu6);
		REGISTE(operation_relu6);
	}
}