#include "reshape.hpp"
#include <memory>
#include <algorithm>
#include <iostream>

namespace glasssix
{
	namespace excalibur
	{
		reshape::reshape(int dim1, int dim2, int dim3, int dim4)
		{
			shape_param_.push_back(dim1);
			shape_param_.push_back(dim2);
			shape_param_.push_back(dim3);
			shape_param_.push_back(dim4);
		}

		reshape::~reshape()
		{

		}

		void reshape::Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			std::vector<int> bottom_shape = bottom->data_shape();
			std::vector<int> top_shape(bottom_shape.size());
			CHECK_EQ(bottom_shape.size(), shape_param_.size());

			bool is_infer = false;
			int infer_axis;
			int mult = 1;
			for (int i = 0; i < shape_param_.size(); i++)
			{
				if (shape_param_[i] == 0)//retain
				{
					top_shape[i] = bottom_shape[i];
					mult *= bottom_shape[i];
				}
				else if (shape_param_[i] == -1)//infer
				{
					if (!is_infer)
					{
						is_infer = true;
						infer_axis = i;
					}
					else
					{
						std::cerr << "at most a single (1) value of -1 may be specified" << std::endl;
						exit(-1);
					}
				}
				else//assigned dimension
				{
					top_shape[i] = shape_param_[i];
					mult *= shape_param_[i];
				}
			}

			int count = bottom->count();
			CHECK_EQ(count % mult, 0);
			top_shape[infer_axis] = count / mult;

			top.reset(new tensor<float>(top_shape, bottom->device(), bottom->order()));

			if (device_ < 0)
			{
				memcpy(top->mutable_cpu_data(), bottom->cpu_data(), count * sizeof(float));
			}
			else
			{
#ifdef USE_CUDA
				cudaMemcpy(top->mutable_gpu_data(), bottom->gpu_data(), count * sizeof(float), cudaMemcpyDefault);
#else
				NO_GPU;
#endif // USE_CUDA

			}
		}
	}
}