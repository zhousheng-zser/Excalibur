#include "tensor.hpp"

namespace excalibur
{
	template <typename Dtype>
	tensor<Dtype>::tensor(tensorType Ttype = NCHW)
	{
		count_ = 0;
		shape_ = std::vector<int>{ 0 };
		device_ = -1;
		data_ = nullptr;
		type_ = Ttype;
	}

	template <typename Dtype>
	tensor<Dtype>::tensor(const std::vector<int>& shape, int device, tensorType Ttype = NCHW)
	{
		count_ = 1;
		for (int i = 0; i < shape.size(); i++)
		{
			count_ *= shape[i];
			shape_.push_back(shape[i]);
		}
		device_ = device;
		data_.reset(new syncedmem(count_ * sizeof(Dtype), device_));
		type_ = Ttype;
	}

	template <typename Dtype>
	tensor<Dtype>::tensor(const int shape, int device = -1, tensorType Ttype = NCHW)
	{
		count_ = shape;
		shape_.push_back(shape);
		device_ = device;
		data_.reset(new syncedmem(count_ * sizeof(Dtype), device_));
		type_ = Ttype;
	}

	template <typename Dtype>
	tensor<Dtype>::tensor(const tensor& t)
	{
		count_ = t.count_;
		device_ = t.device_;
		shape_ = t.shape_;
		this->data_ = t.data_;
		type_ = t.type_;
	}

	template <typename Dtype>
	tensor<Dtype>& tensor<Dtype>::operator=(const tensor& t)
	{
		if (this == &t)
		{
			return *this;
		}
		count_ = t.count_;
		device_ = t.device_;
		shape_ = t.shape_;
		this->data_ = t.data_;
		type_ = t.type_;
		return *this;
	}


	template <typename Dtype>
	tensor<Dtype>::~tensor()
	{
		
	}

	template <typename Dtype>
	tensor<Dtype> tensor<Dtype>::clone() const
	{
		if (empty())
		{
			return tensor();
		}
		tensor t(shape_, device_, type_);
		if (device_ >= 0)
		{
			math_functions::excalibur_copy(count_, this->gpu_data(), t.mutable_gpu_data(), device_);
		}
		else
		{
			math_functions::excalibur_copy(count_, this->cpu_data(), t.mutable_cpu_data(), device_);
		}
		return t;
	}

	template <typename Dtype>
	bool tensor<Dtype>::empty() const
	{
		return data_ == nullptr || count() == 0;
	}

	template <typename Dtype>
	tensor<Dtype> tensor<Dtype>::channel_tensor_ptr(int c)
	{
		if (c >= channels()) 
		{
			LOG(ERROR) << "channel out of index.";
		}

		if (type_ == NHWC)
		{
			tensor t(std::vector<int>{1, height(), width(), 1}, device_, type_);
			if (device_ >= 0)
			{
				const Dtype* s_data = this->gpu_data();
				Dtype* t_data = t.mutable_gpu_data();

				for (int row = 0; row < height(); ++row)
				{
					for (int col = 0; col < width(); ++col)
					{
						t_data[row * width() + col] = s_data[(row * width() + col) * channels() + c];
					}
				}
			}
			else
			{
				const Dtype* s_data = this->cpu_data();
				Dtype* t_data = t.mutable_cpu_data();

				for (int row = 0; row < height(); ++row)
				{
					for (int col = 0; col < width(); ++col)
					{
						t_data[row * width() + col] = s_data[(row * width() + col) * channels() + c];
					}
				}
			}

			return t;
		}
		else
		{
			tensor t(std::vector<int>{1, 1, height(), width()}, device_, type_);
			int offset = height() * width();
			if (device_ >= 0)
			{
				math_functions::excalibur_copy(offset, this->gpu_data() + offset * c, t.mutable_gpu_data(), device_);
			}
			else
			{
				math_functions::excalibur_copy(offset, this->cpu_data() + offset * c, t.mutable_cpu_data(), device_);
			}

			return t;
		}
	}


	template <typename Dtype>
	const Dtype* tensor<Dtype>::cpu_data() const
	{
		CHECK(data_);
		return static_cast<const Dtype*>(data_->cpu_data());
	}

	template <typename Dtype>
	const Dtype* tensor<Dtype>::gpu_data() const
	{
		CHECK(data_);
		return static_cast<const Dtype*>(data_->gpu_data());
	}

	template <typename Dtype>
	Dtype* tensor<Dtype>::mutable_cpu_data() const
	{
		CHECK(data_);
		return static_cast<Dtype*>(data_->mutable_cpu_data());
	}

	template <typename Dtype>
	Dtype* tensor<Dtype>::mutable_gpu_data() const
	{
		CHECK(data_);
		return static_cast<Dtype*>(data_->mutable_gpu_data());
	}

	template <typename Dtype>
	void tensor<Dtype>::set_cpu_data(Dtype* data)
	{
		CHECK(data);
		// Make sure CPU and GPU sizes remain equal
		size_t size = count_ * sizeof(Dtype);
		if (data_->size() != size) {
			data_.reset(new syncedmem(size));
		}
		data_->set_cpu_data(data);
	}

	template <typename Dtype>
	void tensor<Dtype>::set_gpu_data(Dtype* data)
	{
		CHECK(data);
		// Make sure CPU and GPU sizes remain equal
		size_t size = count_ * sizeof(Dtype);
		if (data_->size() != size) {
			data_.reset(new syncedmem(size));
		}
		data_->set_gpu_data(data);
	}

	template class tensor<float>;
	template class tensor<int>;
	template class tensor<char>;
	template class tensor<unsigned char>;
	template class tensor<unsigned int>;
}