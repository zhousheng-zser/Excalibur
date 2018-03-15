#include "cudnn_convolution.hpp"
#ifdef USE_CUDNN
#include <algorithm>

namespace excalibur
{
	// Set to three for the benefit of the backward pass, which
	// can use separate streams for calculating the gradient w.r.t.
	// bias, filter weights, and bottom data for each group independently
#define CUDNN_STREAMS_PER_GROUP 3

	cudnn_convolution::cudnn_convolution(int input_Channel, int output_Channel, int kernelSize,
		int stride, int pad, bool bias_term, int device)
	{
		// Initalize conv params
		num_input_ = input_Channel;
		channels_ = num_input_;
		num_output_ = output_Channel;
		kernel_size_ = kernelSize;
		stride_ = stride;
		pad_ = pad;
		group_ = 1;
		bias_term_ = bias_term;
		device_ = device;

		weights_ = new tensor(std::vector<int>{num_input_*num_output_*kernel_size_*kernel_size_}, device_);
		bias_ = new tensor(std::vector<int>{num_output_}, device_);

		stream_ = new cudaStream_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
		handle_ = new cudnnHandle_t[this->group_ * CUDNN_STREAMS_PER_GROUP];

		// Initialize algorithm arrays
		fwd_algo_ = new cudnnConvolutionFwdAlgo_t[1];

		// initialize size arrays
		workspace_fwd_sizes_ = new size_t[1];

		// workspace data
		workspaceData = NULL;
		workspace = new void*[this->group_ * CUDNN_STREAMS_PER_GROUP];

		for (size_t i = 0; i < 1; ++i) {
			// initialize all to default algorithms
			fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
			// default algorithms don't require workspace
			workspace_fwd_sizes_[i] = 0;
		}

		for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
			CUDA_CHECK(cudaStreamCreate(&stream_[g]));
			CUDNN_CHECK(cudnnCreate(&handle_[g]));
			CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
			workspace[g] = NULL;
		}

		// Set the indexing parameters.
		bias_offset_ = (this->num_output_ / this->group_);

		const int kernel_h = kernel_size_;
		const int kernel_w = kernel_size_;
		cudnn::createFilterDesc(&filter_desc_,
			this->num_output_ / this->group_, this->channels_ / this->group_,
			kernel_h, kernel_w);

		// Create tensor descriptor(s) for data and corresponding convolution(s).
		for (int i = 0; i < 1; i++) {
			cudnnTensorDescriptor_t bottom_desc;
			cudnn::createTensor4dDesc(&bottom_desc);
			bottom_descs_.push_back(bottom_desc);
			cudnnTensorDescriptor_t top_desc;
			cudnn::createTensor4dDesc(&top_desc);
			top_descs_.push_back(top_desc);
			cudnnConvolutionDescriptor_t conv_desc;
			cudnn::createConvolutionDesc(&conv_desc);
			conv_descs_.push_back(conv_desc);
		}

		// Tensor descriptor for bias.
		if (this->bias_term_) {
			cudnn::createTensor4dDesc(&bias_desc_);
		}

		handles_setup_ = true;
	}


	cudnn_convolution::~cudnn_convolution()
	{
		delete weights_;
		delete bias_;
	}

	void cudnn_convolution::set_weights(float* weights)
	{
		weights_->set_gpu_data(weights);
	}

	void cudnn_convolution::set_bias(float* bias)
	{
		bias_->set_gpu_data(bias);
	}

	void cudnn_convolution::Forward_cudnn_gpu(const std::shared_ptr<tensor>& bottom, std::shared_ptr<tensor>& top)
	{
		// calcu output parms
		const int height = bottom->height();
		const int width = bottom->width();
		const int num = bottom->num();
		int height_out = (height + 2 * pad_ - kernel_size_) / stride_ + 1;
		int width_out = (width + 2 * pad_ - kernel_size_) / stride_ + 1;
		top.reset(new tensor(std::vector<int>{num, this->num_output_, height_out, width_out}, this->device_));
		out_spatial_dim_ = width_out*height_out;

		// Specify workspace limit for kernels directly until we have a
		// planning strategy and a rewrite of Caffe's GPU memory mangagement
		size_t workspace_limit_bytes = 8 * 1024 * 1024;

		for (int i = 0; i < 1; i++) {
			cudnn::setTensor4dDesc(&bottom_descs_[i],
				num,
				this->channels_ / this->group_, height, width,
				this->channels_ * height * width,
				height * width, width, 1);
			cudnn::setTensor4dDesc(&top_descs_[i],
				num,
				this->num_output_ / this->group_, height_out, width_out,
				this->num_output_ * this->out_spatial_dim_,
				this->out_spatial_dim_, width_out, 1);
			cudnn::setConvolutionDesc(&conv_descs_[i], bottom_descs_[i],
				filter_desc_, pad_, pad_,
				stride_, stride_);

			// choose forward and backward algorithms + workspace(s)
			CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[0],
				bottom_descs_[i],
				filter_desc_,
				conv_descs_[i],
				top_descs_[i],
				CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
				workspace_limit_bytes,
				&fwd_algo_[i]));

			CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[0],
				bottom_descs_[i],
				filter_desc_,
				conv_descs_[i],
				top_descs_[i],
				fwd_algo_[i],
				&(workspace_fwd_sizes_[i])));
		}

		// reduce over all workspace sizes to get a maximum to allocate / reallocate
		size_t total_workspace_fwd = 0;

		for (size_t i = 0; i < 1; i++) {
			total_workspace_fwd = std::max(total_workspace_fwd, workspace_fwd_sizes_[i]);
		}
		// get max over all operations
		size_t max_workspace = total_workspace_fwd;
		// ensure all groups have enough workspace
		size_t total_max_workspace = max_workspace * (this->group_ * CUDNN_STREAMS_PER_GROUP);

		// free the existing workspace and allocate a new (larger) one
		size_t workspaceSizeInBytes = total_max_workspace;
		int size = (workspaceSizeInBytes) / sizeof(float) + 1;
		workspaceDataBlob = new tensor(std::vector<int>{1, size}, device_);
		workspaceData = workspaceDataBlob->mutable_gpu_data();

		// if we succeed in the allocation, set pointer aliases for workspaces
		for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
			workspace[g] = reinterpret_cast<char *>(workspaceData) + g*max_workspace;
		}

		// Tensor descriptor for bias.
		if (this->bias_term_) {
			cudnn::setTensor4dDesc(&bias_desc_,
				1, this->num_output_ / this->group_, 1, 1);
		}

		const float* weight = this->weights_->gpu_data();
		const float* bias_data = this->bias_->gpu_data();
		const float* bottom_data = bottom->gpu_data();
		float* top_data = top->mutable_gpu_data();

		// Forward through cuDNN in parallel over groups.
		for (int g = 0; g < this->group_; g++) {
			// Filters.
			CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
				cudnn::dataType<float>::one,
				bottom_descs_[0], bottom_data + bottom_offset_ * g,
				filter_desc_, weight + this->weight_offset_ * g,
				conv_descs_[0],
				fwd_algo_[0], workspace[g], workspace_fwd_sizes_[0],
				cudnn::dataType<float>::zero,
				top_descs_[0], top_data + top_offset_ * g));

			// Bias.
			if (this->bias_term_) {
				CUDNN_CHECK(cudnnAddTensor(handle_[g],
					cudnn::dataType<float>::one,
					bias_desc_, bias_data + bias_offset_ * g,
					cudnn::dataType<float>::one,
					top_descs_[0], top_data + top_offset_ * g));
			}
		}
	}

}
#endif // USE_CUDNN