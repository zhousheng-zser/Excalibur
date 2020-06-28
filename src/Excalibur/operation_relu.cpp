#include "../../include/Excalibur/operation_relu.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_relu<Dtype>::operation_relu(const operation_param& param) : operation<Dtype>(param)
		{
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					slope_ = atof(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "-23330")
				{
					//do nothing
				}
				else
				{
					LOG(FATAL) << "Un-supported ReLU Attribution " << split_string(attrs[i], "=")[0];
				}
			}
			params_.inplace_ = true;
		}

		template<typename Dtype>
		void operation_relu<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), tops.size());
			for (size_t i = 0; i < bottoms.size(); i++)
			{
				tops[i].reset(new memory::tensor<float>(bottoms[i]->data_shape(), bottoms[i]->device(), bottoms[i]->order(), bottoms[i]->allocator()));
				float* top_data = tops[i]->mutable_cpu_data();
				const float* bottom_data = bottoms[i]->cpu_data();
				for (size_t j = 0; j < bottoms[i]->count(); j++)
				{
					top_data[j] = bottom_data[j] >= 0.0f ? bottom_data[j] : 0.0f;
				}
			}
		}


		template<typename Dtype>
		void operation_relu<Dtype>::forward_gpu_f32(
#ifdef USE_CUDA
			cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			NOT_IMPLEMENTED;
		}

		INSTANCE_CLASS(operation_relu);
		REGISTE(operation_relu);
	}
}