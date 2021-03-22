#include "../../include/Excalibur/operation_crop.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"
#include "../../include/Excalibur/math_functions.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_crop<Dtype>::operation_crop(const operation_param& param) : operation<Dtype>(param)
		{
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					woffset_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "1")
				{
					hoffset_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "2")
				{
					coffset_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "3")
				{
					outw_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "4")
				{
					outh_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "5")
				{
					outc_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "6")
				{
					woffset2_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "7")
				{
					hoffset2_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "8")
				{
					coffset2_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "9")
				{
					//starts_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "10")
				{
					//ends_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "11")
				{
					//axes_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "-23330")
				{
					//do nothing
				}
				else
				{
					LOG(FATAL) << "Un-supported Crop Attribution " << split_string(attrs[i], "=")[0];
				}
			}
		}

		template<typename Dtype>
		void operation_crop<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
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
			auto top_data = tops[0]->mutable_cpu_data();
			auto bottom_data = bottoms[0]->cpu_data();
			if (bottoms[1]->order() == memory::NCHW)
			{
				int bottom_offset_n = bottoms[0]->count(1, 4);
				int top_offset_n = tops[0]->count(1, 4);

				int bottom_offset_c = bottoms[0]->count(2, 4);
				int top_offset_c = tops[0]->count(2, 4);

				int bottom_offset_h = bottoms[0]->count(3, 4);
				int top_offset_h = tops[0]->count(3, 4);

				for (size_t n = 0; n < bottoms[0]->num(); n++)
				{
					for (size_t c = 0; c < bottoms[1]->channels(); c++)
					{
						for (size_t h = 0; h < bottoms[1]->height(); h++)
						{
							memcpy(top_data + top_offset_n * n + top_offset_c * c + top_offset_h * h,
								bottom_data + bottom_offset_n * n + bottom_offset_c * (coffset_ + c) + bottom_offset_h * (hoffset_ + h) + woffset_, 
								bottoms[1]->width() * sizeof(float));
						}
					}
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

#ifndef USE_CUDA
		STUB_GPU(operation_crop);
#endif

		INSTANCE_CLASS(operation_crop);
		REGISTE(operation_crop);
	}
}