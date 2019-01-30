#include "eltwise.hpp"
#include <algorithm>

namespace glasssix
{
	namespace excalibur
	{
		eltwise::eltwise(int type, int device)
		{
			type_ = (eltwise_type)type;
			device_ = device;
		}

		eltwise::~eltwise()
		{
		}

		void eltwise::Forward_cpu(const std::vector<std::shared_ptr<tensor<float>>> bottom, std::shared_ptr<tensor<float>>& top)
		{
			coeffs_ = std::vector<float>(bottom.size(), 1);
			for (int i = 1; i < bottom.size(); ++i) {
				CHECK(bottom[i]->data_shape() == bottom[0]->data_shape());
			}
			top.reset(new tensor<float>(bottom[0]->data_shape(), device_));
			//
			const float* bottom_data_a = nullptr;
			const float* bottom_data_b = nullptr;
			const int count = (top)->count(0, (top)->data_shape().size());
			float* top_data = (top)->mutable_cpu_data();
			switch (type_)
			{
			case SUM:
				memset(top_data, 0, count * sizeof(float));
				for (int i = 0; i < bottom.size(); ++i)
				{
					cblas_saxpy(count, coeffs_[i], bottom[i]->cpu_data(), 1, top_data, 1);
				}
				break;
			case MAX:
				memset(top_data, static_cast<float>(-FLT_MAX), count * sizeof(float));
				bottom_data_a = bottom[0]->cpu_data();
				bottom_data_b = bottom[1]->cpu_data();
				for (int idx = 0; idx < count; ++idx) {
					top_data[idx] = std::max(bottom_data_a[idx], bottom_data_b[idx]);
				}
				// bottom 2++
				for (int blob_idx = 2; blob_idx < bottom.size(); ++blob_idx) {
					bottom_data_b = bottom[blob_idx]->cpu_data();
					for (int idx = 0; idx < count; ++idx) {
						top_data[idx] = std::max(top_data[idx], bottom_data_b[idx]);
					}
				}
				break;
			default:
				LOG(FATAL) << "Unknown elementwise operation.";
			}
		}

	}
}

