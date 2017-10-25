#include "cudnn_convolution.hpp"
#ifdef USE_CUDNN
#include <algorithm>

namespace excalibur
{


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

		// Initialize CUDA streams and cuDNN.
		stream_ = new cudaStream_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
		handle_ = new cudnnHandle_t[this->group_ * CUDNN_STREAMS_PER_GROUP];

		// Initialize algorithm arrays
		fwd_algo_ = new cudnnConvolutionFwdAlgo_t[1];

		// initialize size arrays
		workspace_fwd_sizes_ = new size_t[1];

		// workspace data
		workspaceSizeInBytes = 0;
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

		// Create filter descriptor.
		//const int* kernel_shape_data = this->kernel_shape_->cpu_data();
		const int kernel_h = kernel_size_/*kernel_shape_data[0]*/;
		const int kernel_w = kernel_size_/*kernel_shape_data[1]*/;
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
		// Check that handles have been setup before destroying.
		if (!handles_setup_) { return; }

		for (int i = 0; i < bottom_descs_.size(); i++) {
			cudnnDestroyTensorDescriptor(bottom_descs_[i]);
			cudnnDestroyTensorDescriptor(top_descs_[i]);
			cudnnDestroyConvolutionDescriptor(conv_descs_[i]);
		}
		if (this->bias_term_) {
			cudnnDestroyTensorDescriptor(bias_desc_);
		}
		cudnnDestroyFilterDescriptor(filter_desc_);

		for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
			cudaStreamDestroy(stream_[g]);
			cudnnDestroy(handle_[g]);
		}

		delete[] stream_;
		delete[] handle_;
		delete[] fwd_algo_;
		delete[] workspace_fwd_sizes_;
	}

	void cudnn_convolution::pre_Forward_cudnn_gpu(const std::shared_ptr<tensor>& bottom, std::shared_ptr<tensor>& top)
	{
		// calcu output parms
		int height_out = (bottom->data_shape()[2] + 2 * pad_ - kernel_size_) / stride_ + 1;
		int width_out = (bottom->data_shape()[3] + 2 * pad_ - kernel_size_) / stride_ + 1;
		const int height = bottom->height();
		const int width = bottom->width();
		this->num_ = bottom->num();
		out_spatial_dim_ = width_out*height_out;

		bottom_dim_ = bottom->count(1, 4);
		bottom_offset_ = this->bottom_dim_ / this->group_;
		top.reset(new tensor(std::vector<int>{num_, num_output_, height_out, width_out}, device_));
		top_dim_ = top->count(1, 4);
		top_offset_ = this->top_dim_ / this->group_;
		kernel_dim_ = num_input_*kernel_size_*kernel_size_;
		weight_offset_ = num_output_*kernel_dim_ / group_;
		// Specify workspace limit for kernels directly until we have a
		// planning strategy and a rewrite of Caffe's GPU memory mangagement
		size_t workspace_limit_bytes = 8 * 1024 * 1024;

		for (int i = 0; i < 1; i++) {
			cudnn::setTensor4dDesc(&bottom_descs_[i],
				this->num_,
				this->channels_ / this->group_, height, width,
				this->channels_ * height * width,
				height * width, width, 1);
			cudnn::setTensor4dDesc(&top_descs_[i],
				this->num_,
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
			total_workspace_fwd = std::max(total_workspace_fwd,
				workspace_fwd_sizes_[i]);
		}

		// ensure all groups have enough workspace
		size_t total_max_workspace = total_workspace_fwd *
			(this->group_ * CUDNN_STREAMS_PER_GROUP);

		//DLOG(INFO) << "Reallocating workspace storage: " << total_max_workspace;
		workspaceSizeInBytes = total_max_workspace;

		// free the existing workspace and allocate a new (larger) one
		int size = (workspaceSizeInBytes) / sizeof(float) + 1;
		workspaceDataBlob.reset(new tensor(std::vector<int>{size}, device_));
		workspaceData = workspaceDataBlob->mutable_gpu_data();

		// if we succeed in the allocation, set pointer aliases for workspaces
		for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
			workspace[g] = reinterpret_cast<char *>(workspaceData) + g*total_workspace_fwd;
		}

		// Tensor descriptor for bias.
		if (this->bias_term_) {
			cudnn::setTensor4dDesc(&bias_desc_,
				1, this->num_output_ / this->group_, 1, 1);
		}
	}
}
#endif // USE_CUDNN