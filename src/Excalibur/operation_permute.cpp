#include "../../include/Excalibur/operation_permute.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_permute<Dtype>::operation_permute(const operation_param& param) : operation<Dtype>(param)
		{
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					permute_type_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else
				{
					LOG(FATAL) << "Un-supported Permute Attribution " << split_string(attrs[i], "=")[0];
				}
			}
		}

		template<typename Dtype>
		void operation_permute<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_GE(bottoms.size(), 1);
			CHECK_EQ(tops.size(), 1);
			bool convert_order = false;
			if (bottoms[0]->order() != memory::NCHW)
			{
				bottoms[0]->convert_order();
				convert_order = true;
			}
			if (permute_type_ == 0)
			{
				NOT_IMPLEMENTED;
			}
			else if (permute_type_ == 1)
			{
				NOT_IMPLEMENTED;
			}
			else if (permute_type_ == 2)
			{
				NOT_IMPLEMENTED;
			}
			else if (permute_type_ == 3)
			{
				//axis order: 0 2 3 1(NHWC)
				tops[0].reset(new memory::tensor<float>(std::vector<int>{bottoms[0]->num(), bottoms[0]->height(), bottoms[0]->width(), bottoms[0]->channels()},
					-1, memory::NCHW, bottoms[0]->allocator()));
				const float* bottom_data = bottoms[0]->cpu_data();
				float* top_data = tops[0]->mutable_cpu_data();
				const int channels = bottoms[0]->channels();
				for (size_t n = 0; n < bottoms[0]->num(); n++)
				{
					const int bottom_offset_n = bottoms[0]->count(1, 4) * n;
					for (size_t c = 0; c < bottoms[0]->channels(); c++)
					{
						const int bottom_offset_c = bottoms[0]->count(2, 4) * c;
						for (size_t h = 0; h < bottoms[0]->height(); h++)
						{
							const int bottom_offset_h = bottoms[0]->count(3, 4) * h;
							for (size_t w = 0; w < bottoms[0]->width(); w++)
							{
								top_data[bottom_offset_n + (bottom_offset_h + w) * channels + c] =
									bottom_data[bottom_offset_n + bottom_offset_c + bottom_offset_h + w];
							}
						}
					}
				}
			}
			else if (permute_type_ == 4)
			{
				NOT_IMPLEMENTED;
			}
			else if (permute_type_ == 5)
			{
				NOT_IMPLEMENTED;
			}
			else
			{
				LOG(FATAL) << "Un-supported permute type.";
			}
			if (convert_order)
			{
				tops[0]->convert_order();
			}
		}


		template<typename Dtype>
		void operation_permute<Dtype>::forward_gpu_f32(
#ifdef USE_CUDA
			cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			NOT_IMPLEMENTED;
		}

		INSTANCE_CLASS(operation_permute);
		REGISTE(operation_permute);
	}
}