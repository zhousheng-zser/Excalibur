#include "ImageTensor.hpp"

namespace excalibur
{
	template <typename Dtype>
	ImageTensor<Dtype>::ImageTensor(const std::vector<int>& shape, int device)
	{
		count_ = 1;
		for (int i = 0; i < shape.size(); i++)
		{
			count_ *= shape[i];
			shape_.push_back(shape[i]);
		}
		device_ = device;
		data_ = new syncedmem(count_ * sizeof(Dtype), device_);
	}

	template <typename Dtype>
	ImageTensor<Dtype>::ImageTensor(int img_width, int img_height, int32_t img_num_channels, int device)
	{
		count_ = 1 * img_width * img_height * img_num_channels;
		shape_.push_back(1);
		shape_.push_back(img_num_channels);
		shape_.push_back(img_height);
		shape_.push_back(img_width);
		device_ = device;
		data_ = new syncedmem(count_ * sizeof(Dtype), device_);
	}


	template <typename Dtype>
	ImageTensor<Dtype>::~ImageTensor()
	{
		delete data_;
	}

	template <typename Dtype>
	const Dtype* ImageTensor<Dtype>::cpu_data() const
	{
		CHECK(data_);
		return static_cast<const Dtype*>(data_->cpu_data());
	}

	template <typename Dtype>
	const Dtype* ImageTensor<Dtype>::gpu_data() const
	{
		CHECK(data_);
		return static_cast<const Dtype*>(data_->gpu_data());
	}

	template <typename Dtype>
	Dtype* ImageTensor<Dtype>::mutable_cpu_data() const
	{
		CHECK(data_);
		return static_cast<Dtype*>(data_->mutable_cpu_data());
	}

	template <typename Dtype>
	Dtype* ImageTensor<Dtype>::mutable_gpu_data() const
	{
		CHECK(data_);
		return static_cast<Dtype*>(data_->mutable_gpu_data());
	}

	template <typename Dtype>
	void ImageTensor<Dtype>::set_cpu_data(Dtype* data)
	{
		CHECK(data);
		// Make sure CPU and GPU sizes remain equal
		size_t size = count_ * sizeof(Dtype);
		if (data_->size() != size) {
			data_ = new syncedmem(size);
			//data_.reset(new SyncedMemory(size));
		}
		data_->set_cpu_data(data);
	}

	template <typename Dtype>
	void ImageTensor<Dtype>::set_gpu_data(Dtype* data)
	{
		CHECK(data);
		// Make sure CPU and GPU sizes remain equal
		size_t size = count_ * sizeof(Dtype);
		if (data_->size() != size) {
			data_ = new syncedmem(size);
			//data_.reset(new SyncedMemory(size));
		}
		data_->set_gpu_data(data);
	}

	template class ImageTensor<float>;
	template class ImageTensor<int>;
	template class ImageTensor<unsigned char>;
}


