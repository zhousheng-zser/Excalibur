#include "cudnn_convolution.hpp"
#ifdef USE_CUDNN

namespace excalibur
{
	void cudnn_convolution::Forward_cudnn_gpu(const std::shared_ptr<tensor>& bottom, std::shared_ptr<tensor>& top)
	{
		this->pre_Forward_cudnn_gpu(bottom, top);
		const float* weight = this->weights_->gpu_data();
		const float* bias_data = this->bias_->gpu_data();
		for (int i = 0; i < 1; ++i)
		{
			const float* bottom_data = bottom->gpu_data();
			float* top_data = top->mutable_gpu_data();
			// Forward through cuDNN in parallel over groups.
			for (int g = 0; g < this->group_; g++)
			{
				// Filters.
				CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
					cudnn::dataType<float>::one,
					bottom_descs_[i], bottom_data + bottom_offset_ * g,
					filter_desc_, weight + this->weight_offset_ * g,
					conv_descs_[i],
					fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
					cudnn::dataType<float>::zero,
					top_descs_[i], top_data + top_offset_ * g));

				// Bias.
				if (this->bias_term_) {
					CUDNN_CHECK(cudnnAddTensor(handle_[g],
						cudnn::dataType<float>::one,
						bias_desc_, bias_data + bias_offset_ * g,
						cudnn::dataType<float>::one,
						top_descs_[i], top_data + top_offset_ * g));
				}
			}
		}
	}
}

#endif