#include "../../include/Excalibur/operation_reshape.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_reshape<Dtype>::operation_reshape(const operation_param& param) : operation<Dtype>(param)
		{
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					w_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "1")
				{
					h_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "2")
				{
					c_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "3")
				{
					permute_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "-23330")
				{
					//do nothing
				}
				else
				{
					LOG(FATAL) << "Un-supported Reshape Attribution " << split_string(attrs[i], "=")[0];
				}
			}
			params_.inplace_ = true;
		}

		template<typename Dtype>
		void operation_reshape<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			CHECK_EQ(tops.size(), 1);
			std::vector<int> shape = bottoms[0]->data_shape();
			if (bottoms[0]->order() == memory::NCHW)
			{
				if (n_ > 0)
				{
					shape[0] = n_;
				}
				if (c_ > 0)
				{
					shape[1] = c_;
				}
				if (h_ > 0)
				{
					shape[2] = h_;
				}
				if (w_ > 0)
				{
					shape[3] = w_;
				}

				if (n_ < 0)
				{
					CHECK_EQ(bottoms[0]->count() % (shape[1] * shape[2] * shape[3]), 0);
					shape[0] = bottoms[0]->count() / (shape[1] * shape[2] * shape[3]);
				}
				if (c_ < 0)
				{
					CHECK_EQ(bottoms[0]->count() % (shape[0] * shape[2] * shape[3]), 0);
					shape[1] = bottoms[0]->count() / (shape[0] * shape[2] * shape[3]);
				}
				if (h_ < 0)
				{
					CHECK_EQ(bottoms[0]->count() % (shape[1] * shape[0] * shape[3]), 0);
					shape[2] = bottoms[0]->count() / (shape[1] * shape[0] * shape[3]);
				}
				if (w_ < 0)
				{
					CHECK_EQ(bottoms[0]->count() % (shape[1] * shape[2] * shape[0]), 0);
					shape[3] = bottoms[0]->count() / (shape[1] * shape[2] * shape[0]);
				}
			}
			else if (bottoms[0]->order() == memory::NHWC)
			{
				if (n_ > 0)
				{
					shape[0] = n_;
				}
				if (c_ > 0)
				{
					shape[3] = c_;
				}
				if (h_ > 0)
				{
					shape[1] = h_;
				}
				if (w_ > 0)
				{
					shape[2] = w_;
				}

				if (n_ < 0)
				{
					CHECK_EQ(bottoms[0]->count() % (shape[1] * shape[2] * shape[3]), 0);
					shape[0] = bottoms[0]->count() / (shape[1] * shape[2] * shape[3]);
				}
				if (c_ < 0)
				{
					CHECK_EQ(bottoms[0]->count() % (shape[0] * shape[2] * shape[1]), 0);
					shape[3] = bottoms[0]->count() / (shape[0] * shape[2] * shape[1]);
				}
				if (h_ < 0)
				{
					CHECK_EQ(bottoms[0]->count() % (shape[2] * shape[0] * shape[3]), 0);
					shape[1] = bottoms[0]->count() / (shape[2] * shape[0] * shape[3]);
				}
				if (w_ < 0)
				{
					CHECK_EQ(bottoms[0]->count() % (shape[1] * shape[3] * shape[0]), 0);
					shape[2] = bottoms[0]->count() / (shape[1] * shape[3] * shape[0]);
				}
			}
			else
			{
				LOG(FATAL) << "Un-suppoted order type.";
			}
			bottoms[0]->reshape(shape);
			tops[0] = bottoms[0];
		}


//		template<typename Dtype>
//		void operation_reshape<Dtype>::forward_gpu_f32(
//#ifdef USE_CUDA
//			cublasHandle_t &cublas_handle_,
//#ifdef USE_CUDNN
//			cudnnHandle_t cudnn_handle,
//#endif //!USE_CUDNN
//#endif //!USE_CUDA
//			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
//			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
//		{
//			NOT_IMPLEMENTED;
//		}

		INSTANCE_CLASS(operation_reshape);
		REGISTE(operation_reshape);
	}
}