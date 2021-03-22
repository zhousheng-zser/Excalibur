#include "../../include/Excalibur/operation_concat.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"
#include "../../include/Excalibur/math_functions.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_concat<Dtype>::operation_concat(const operation_param& param) : operation<Dtype>(param)
		{
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					axis_ = atof(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "-23330")
				{
					//do nothing
				}
				else
				{
					LOG(FATAL) << "Un-supported Concat Attribution " << split_string(attrs[i], "=")[0];
				}
			}
		}

		template<typename Dtype>
		void operation_concat<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_GE(bottoms.size(), 2);
			CHECK_EQ(tops.size(), 1);
			//Accroding to NCNN protocol, axis start in domension c(CHW), when write to param file, it -1 without any reason.
			//So we should +2 to restore the Correct axis in NCHW/NHWC
			int concat_axis = axis_ + 2;
			int concat_axis_dim = 0;
			if (bottoms[0]->order() == memory::NCHW)
			{
				for (size_t i = 0; i < bottoms.size(); i++)
				{
					concat_axis_dim += bottoms[i]->data_shape()[concat_axis];
				}
			}
			else if (bottoms[0]->order() == memory::NHWC)
			{
				if (concat_axis == 1)
				{
					concat_axis = 3;
				}
				if (concat_axis == 2)
				{
					concat_axis = 1;
				}
				if (concat_axis == 3)
				{
					concat_axis = 2;
				}
				for (size_t i = 0; i < bottoms.size(); i++)
				{
					concat_axis_dim += bottoms[i]->data_shape()[concat_axis];
				}
			}
			else
			{
				LOG(FATAL) << "Un-supported order type.";
			}
			std::vector<int> concat_shape(4);
			for (size_t j = 0; j < concat_shape.size(); j++)
			{
				if (j != concat_axis)
				{
					concat_shape[j] = bottoms[0]->data_shape()[j];
				}
				else
				{
					concat_shape[j] = concat_axis_dim;
				}
			}
			tops[0].reset(new memory::tensor<float>(concat_shape, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
			if (bottoms[0]->order() == memory::NCHW)
			{
				float* top_data = tops[0]->mutable_cpu_data();
				if (concat_axis == 1)
				{
					for (size_t n = 0; n < concat_shape[0]; n++)
					{
						int top_offset = 0;
						float* top_data_slice_n = top_data + tops[0]->count(concat_axis, 4) * n;
						for (int i = 0; i < bottoms.size(); i++)
						{
							auto bottom_data_slice_n = bottoms[i]->cpu_data() + bottoms[i]->count(concat_axis, 4) * n;
							math_functions::excalibur_copy(bottoms[i]->count(concat_axis, 4), bottom_data_slice_n,
								top_data_slice_n + top_offset, bottoms[i]->device());
							top_offset += bottoms[i]->count(concat_axis, 4);
						}
					}
				}
				else if (concat_axis == 2)
				{
					for (size_t n = 0; n < concat_shape[0]; n++)
					{
						for (size_t c = 0; c < concat_shape[1]; c++)
						{
							int top_offset = 0;
							float* top_data_slice_c = top_data + tops[0]->count(1, 4) * n + tops[0]->count(2, 4) * c;
							for (int i = 0; i < bottoms.size(); i++)
							{
								auto bottom_data_slice_c = bottoms[i]->cpu_data() + bottoms[i]->count(1, 4) * n + bottoms[i]->count(2, 4) * c;
								math_functions::excalibur_copy(bottoms[i]->count(concat_axis, 4), bottom_data_slice_c,
									top_data_slice_c + top_offset, bottoms[i]->device());
								top_offset += bottoms[i]->count(concat_axis, 4);
							}
						}
					}
					/*for (size_t i = 7000; i < 7100; i++)
					{
						std::cout << top_data[i] << " ";
					}
					std::cout << std::endl;*/
				}
				else
				{
					NOT_IMPLEMENTED;
				}
				/*int num_concats = bottoms[0]->count(0, concat_axis);
				int concat_input_size = bottoms[0]->count(concat_axis + 1, 4);
				int offset_concat_axis = 0;
				const int top_concat_axis = concat_axis_dim;
				for (int i = 0; i < bottoms.size(); ++i)
				{
					const float* bottom_data = bottoms[i]->cpu_data();
					const int bottom_concat_axis = bottoms[i]->data_shape()[concat_axis];
					for (int n = 0; n < num_concats; ++n)
					{
						math_functions::excalibur_copy(bottom_concat_axis * concat_input_size,
							bottom_data + n * bottom_concat_axis * concat_input_size,
							top_data + (n * top_concat_axis + offset_concat_axis) * concat_input_size,
							bottoms[i]->device());
					}
					offset_concat_axis += bottom_concat_axis;
				}*/
			}
			else if (bottoms[0]->order() == memory::NHWC)
			{
				NOT_IMPLEMENTED;
			}
			else
			{
				LOG(FATAL) << "Un-supported order type.";
			}
		}

		template<typename Dtype>
		void operation_concat<Dtype>::forward_gpu_f32(
#ifdef USE_CUDA
			cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_GE(bottoms.size(), 2);
			CHECK_EQ(tops.size(), 1);
			//Accroding to NCNN protocol, axis start in domension c(CHW), when write to param file, it -1 without any reason.
			//So we should +2 to restore the Correct axis in NCHW/NHWC
			int concat_axis = axis_ + 2;
			int concat_axis_dim = 0;
			if (bottoms[0]->order() == memory::NCHW)
			{
				for (size_t i = 0; i < bottoms.size(); i++)
				{
					concat_axis_dim += bottoms[i]->data_shape()[concat_axis];
				}
			}
			else if (bottoms[0]->order() == memory::NHWC)
			{
				if (concat_axis == 1)
				{
					concat_axis = 3;
				}
				if (concat_axis == 2)
				{
					concat_axis = 1;
				}
				if (concat_axis == 3)
				{
					concat_axis = 2;
				}
				for (size_t i = 0; i < bottoms.size(); i++)
				{
					concat_axis_dim += bottoms[i]->data_shape()[concat_axis];
				}
			}
			else
			{
				LOG(FATAL) << "Un-supported order type.";
			}
			std::vector<int> concat_shape(4);
			for (size_t j = 0; j < concat_shape.size(); j++)
			{
				if (j != concat_axis)
				{
					concat_shape[j] = bottoms[0]->data_shape()[j];
				}
				else
				{
					concat_shape[j] = concat_axis_dim;
				}
			}
			tops[0].reset(new memory::tensor<float>(concat_shape, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
			if (bottoms[0]->order() == memory::NCHW)
			{
				float* top_data = tops[0]->mutable_gpu_data();
				if (concat_axis == 1)
				{
					for (size_t n = 0; n < concat_shape[0]; n++)
					{
						int top_offset = 0;
						float* top_data_slice_n = top_data + tops[0]->count(concat_axis, 4) * n;
						for (int i = 0; i < bottoms.size(); i++)
						{
							auto bottom_data_slice_n = bottoms[i]->gpu_data() + bottoms[i]->count(concat_axis, 4) * n;
							math_functions::excalibur_copy(bottoms[i]->count(concat_axis, 4), bottom_data_slice_n,
								top_data_slice_n + top_offset, bottoms[i]->device());
							top_offset += bottoms[i]->count(concat_axis, 4);
						}
					}
				}
				else if (concat_axis == 2)
				{
					for (size_t n = 0; n < concat_shape[0]; n++)
					{
						for (size_t c = 0; c < concat_shape[1]; c++)
						{
							int top_offset = 0;
							float* top_data_slice_c = top_data + tops[0]->count(1, 4) * n + tops[0]->count(2, 4) * c;
							for (int i = 0; i < bottoms.size(); i++)
							{
								auto bottom_data_slice_c = bottoms[i]->gpu_data() + bottoms[i]->count(1, 4) * n + bottoms[i]->count(2, 4) * c;
								math_functions::excalibur_copy(bottoms[i]->count(concat_axis, 4), bottom_data_slice_c,
									top_data_slice_c + top_offset, bottoms[i]->device());
								top_offset += bottoms[i]->count(concat_axis, 4);
							}
						}
					}
					/*for (size_t i = 7000; i < 7100; i++)
					{
						std::cout << top_data[i] << " ";
					}
					std::cout << std::endl;*/
				}
				else
				{
					NOT_IMPLEMENTED;
				}
				/*int num_concats = bottoms[0]->count(0, concat_axis);
				int concat_input_size = bottoms[0]->count(concat_axis + 1, 4);
				int offset_concat_axis = 0;
				const int top_concat_axis = concat_axis_dim;
				for (int i = 0; i < bottoms.size(); ++i)
				{
					const float* bottom_data = bottoms[i]->cpu_data();
					const int bottom_concat_axis = bottoms[i]->data_shape()[concat_axis];
					for (int n = 0; n < num_concats; ++n)
					{
						math_functions::excalibur_copy(bottom_concat_axis * concat_input_size,
							bottom_data + n * bottom_concat_axis * concat_input_size,
							top_data + (n * top_concat_axis + offset_concat_axis) * concat_input_size,
							bottoms[i]->device());
					}
					offset_concat_axis += bottom_concat_axis;
				}*/
			}
			else if (bottoms[0]->order() == memory::NHWC)
			{
				NOT_IMPLEMENTED;
			}
			else
			{
				LOG(FATAL) << "Un-supported order type.";
			}
		}

		INSTANCE_CLASS(operation_concat);
		REGISTE(operation_concat);
	}
}