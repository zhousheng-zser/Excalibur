#include "../../include/Excalibur/operation_eltwise.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"
#include "../../include/Excalibur/math_functions.hpp"
#include <algorithm>
#include <cfloat>

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_eltwise<Dtype>::operation_eltwise(const operation_param& param) : operation<Dtype>(param)
		{
			coeffs_ = std::vector<float>(param.input_count_, 1.0f);
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					type_ = (eltwise_type)atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "-23301")
				{
					auto coeff_str = split_string(split_string(attrs[i], "=")[1], ",");
					CHECK_EQ((eltwise_type)type_, eltwise_type::SUM);
					for (size_t j = 1; j < atoi(coeff_str[0].c_str()); j++)
					{
						coeffs_[j - 1] = atof(coeff_str[j].c_str());
					}
				}
				else if (split_string(attrs[i], "=")[0] == "-23330")
				{
					//do nothing
				}
				else
				{
					LOG(FATAL) << "Un-supported Eltwise Attribution " << split_string(attrs[i], "=")[0];
				}
			}
		}

		template<typename Dtype>
		void operation_eltwise<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_GE(bottoms.size(), 2);
			for (int i = 1; i < bottoms.size(); ++i)
			{
				CHECK(bottoms[i]->data_shape() == bottoms[0]->data_shape());
			}
			CHECK_EQ(tops.size(), 1);
			tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
			const float* bottom_data_a = nullptr;
			const float* bottom_data_b = nullptr;
			const int count = tops[0]->count();
			float* top_data = tops[0]->mutable_cpu_data();
			switch (type_)
			{
			case SUM:
				math_functions::cpu_set(count, 0.0f, top_data);
				for (int i = 0; i < bottoms.size(); ++i)
				{
					cblas_saxpby(count, coeffs_[i], bottoms[i]->cpu_data(), 1, 1.0f, top_data, 1);
				}
				break;
			case MAX:
				math_functions::cpu_set(count, static_cast<float>(-FLT_MAX), top_data);
				bottom_data_a = bottoms[0]->cpu_data();
				bottom_data_b = bottoms[1]->cpu_data();
				for (int idx = 0; idx < count; ++idx) 
				{
					top_data[idx] = std::max(bottom_data_a[idx], bottom_data_b[idx]);
				}
				// bottom 2++
				for (int blob_idx = 2; blob_idx < bottoms.size(); ++blob_idx) 
				{
					bottom_data_b = bottoms[blob_idx]->cpu_data();
					for (int idx = 0; idx < count; ++idx) 
					{
						top_data[idx] = std::max(top_data[idx], bottom_data_b[idx]);
					}
				}
				break;
			case PROD:
				math_functions::cpu_mul(count, bottoms[0]->cpu_data(), bottoms[1]->cpu_data(), top_data);
				for (int i = 2; i < bottoms.size(); ++i) 
				{
					math_functions::cpu_mul(count, top_data, bottoms[i]->cpu_data(), top_data);
				}
				break;
			default:
				LOG(FATAL) << "Unknown elementwise operation.";
			}
		}


		template<typename Dtype>
		void operation_eltwise<Dtype>::forward_gpu_f32(
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

		INSTANCE_CLASS(operation_eltwise);
		REGISTE(operation_eltwise);
	}
}