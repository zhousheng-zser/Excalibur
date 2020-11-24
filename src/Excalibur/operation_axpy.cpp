#include "../../include/Excalibur/operation_axpy.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"
#include "../../include/Excalibur/math_functions.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_axpy<Dtype>::operation_axpy(const operation_param& param) : operation<Dtype>(param)
		{
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					//do nothing
				}
				else
				{
					LOG(FATAL) << "Un-supported Axpy Attribution " << split_string(attrs[i], "=")[0];
				}
			}
			this->params_.inplace_ = false;
		}

		template<typename Dtype>
		void operation_axpy<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 2);
			CHECK_EQ(tops.size(), 1);
			CHECK_EQ(bottoms[0]->num(), bottoms[1]->num());
			CHECK_EQ(bottoms[0]->channels(), bottoms[1]->channels());
			CHECK_EQ(bottoms[1]->width(), 1);
			CHECK_EQ(bottoms[1]->height(), 1);
			if (bottoms[0]->order() != memory::NCHW)
			{
				NOT_IMPLEMENTED;
			}
			tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
			int num = bottoms[0]->num();
			int channel = bottoms[0]->channels();
			int spatial_dim = bottoms[0]->count(2, 4);
			auto scale_data = bottoms[1]->cpu_data();
			auto bottom_data = bottoms[0]->cpu_data();
			auto top_data = tops[0]->mutable_cpu_data();
			for (size_t n = 0; n < num; n++)
			{
				for (size_t c = 0; c < channel; c++)
				{
					auto top_data_offset = top_data + n * channel * spatial_dim + c * spatial_dim;
					auto bottom_data_offset = bottom_data + n * channel * spatial_dim + c * spatial_dim;
					for (size_t i = 0; i < spatial_dim; i++)
					{
						top_data_offset[i] = scale_data[c] * bottom_data_offset[i];
					}
				}
			}
		}

#ifndef USE_CUDA
		STUB_GPU(operation_axpy);
#endif

		INSTANCE_CLASS(operation_axpy);
		REGISTE(operation_axpy);
	}
}