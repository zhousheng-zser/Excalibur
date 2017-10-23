#include "tensor.hpp"

namespace excalibur
{
	tensor::tensor(const std::vector<int>& shape, int device)
	{
		count_ = 1;
		for (int i = 0; i < shape.size(); i++)
		{
			count_ *= shape[i];
			shape_.push_back(shape[i]);
		}
		device_ = device;
		data_ = new syncedmem(count_ * sizeof(float), device_);
	}


	tensor::~tensor()
	{
		delete data_;
	}

	const float* tensor::cpu_data() const
	{
		CHECK(data_);
		return static_cast<const float*>(data_->cpu_data());
	}

	const float* tensor::gpu_data() const
	{
		CHECK(data_);
		return static_cast<const float*>(data_->gpu_data());
	}

	float* tensor::mutable_cpu_data() const
	{
		CHECK(data_);
		return static_cast<float*>(data_->mutable_cpu_data());
	}

	float* tensor::mutable_gpu_data() const
	{
		CHECK(data_);
		return static_cast<float*>(data_->mutable_gpu_data());
	}

	void tensor::set_cpu_data(float* data)
	{
		CHECK(data);
		// Make sure CPU and GPU sizes remain equal
		size_t size = count_ * sizeof(float);
		if (data_->size() != size) {
			data_ = new syncedmem(size);
			//data_.reset(new SyncedMemory(size));
		}
		data_->set_cpu_data(data);
	}

	void tensor::set_gpu_data(float* data)
	{
		CHECK(data);
		// Make sure CPU and GPU sizes remain equal
		size_t size = count_ * sizeof(float);
		if (data_->size() != size) {
			data_ = new syncedmem(size);
			//data_.reset(new SyncedMemory(size));
		}
		data_->set_gpu_data(data);
	}

}