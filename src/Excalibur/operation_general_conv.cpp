#include "../../include/Excalibur/operation_general_conv.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_general_conv<Dtype>::operation_general_conv(const operation_param& param) : operation<Dtype>(param)
		{
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					output_channel_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "1")
				{
					kernel_size_w_ = atoi(split_string(attrs[i], "=")[1].c_str());
					kernel_size_h_ = kernel_size_w_;
				}
				else if (split_string(attrs[i], "=")[0] == "2")
				{
					dilation_w_ = atoi(split_string(attrs[i], "=")[1].c_str());
					dilation_h_ = dilation_w_;
				}
				else if (split_string(attrs[i], "=")[0] == "3")
				{
					stride_w_ = atoi(split_string(attrs[i], "=")[1].c_str());
					stride_h_ = stride_w_;
				}
				else if (split_string(attrs[i], "=")[0] == "4")
				{
					pad_left_ = atoi(split_string(attrs[i], "=")[1].c_str());
					pad_right_ = pad_left_;
					pad_top_ = pad_left_;
					pad_bottom_ = pad_left_;
				}
				else if (split_string(attrs[i], "=")[0] == "5")
				{
					bias_term_ = (bool)atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "6")
				{
					weight_data_size_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "7")
				{
					group_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "8")
				{
					int8_scale_term_ = (bool)atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "9")
				{
					activation_type_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "10")
				{
					NOT_IMPLEMENTED;
				}
				else if (split_string(attrs[i], "=")[0] == "11")
				{
					kernel_size_h_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "12")
				{
					dilation_h_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "13")
				{
					stride_h_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "14")
				{
					pad_top_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "15")
				{
					pad_right_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "16")
				{
					pad_bottom_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "17")
				{
					impl_type_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "18")
				{
					//pad_value_ = (Dtype)atof(split_string(attrs[i], "=")[1].c_str());
					output_pad_right_ = atoi(split_string(attrs[i], "=")[1].c_str());
					output_pad_bottom_ = output_pad_right_;
				}
				else if (split_string(attrs[i], "=")[0] == "19")
				{
					output_pad_bottom_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "20")
				{
					output_dim_h_ = atoi(split_string(attrs[i], "=")[1].c_str());
					output_dim_w_ = output_dim_h_;
				}
				else if (split_string(attrs[i], "=")[0] == "21")
				{
					output_dim_w_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "-23330")
				{
					//do nothing
				}
				else
				{
					LOG(FATAL) << "Un-supported Convolution Attribution " << split_string(attrs[i], "=")[0];
				}
			}

			CHECK_EQ(output_channel_ % group_, 0);
			
		}

		INSTANCE_CLASS(operation_general_conv);
	}
}