#include "Pandora_Blob.hpp"

namespace Excalibur
{
	template <typename Dtype>
	Pandora_Blob<Dtype>::Pandora_Blob(const std::vector<int>& shape, int gpu_device, Avalon mode)
	{
		// capacity_ must be initialized before calling Reshape
		capacity_ = 0;
		gpu_device_ = gpu_device;
		mode_ = mode;
		Reshape(shape);
	}

	template <typename Dtype>
	Pandora_Blob<Dtype>::~Pandora_Blob()
	{
		shape_.clear();
		delete data_;
	}

	template <typename Dtype>
	void Pandora_Blob<Dtype>::Reshape(const std::vector<int>& shape)
	{
		CHECK_LE(shape.size(), kMaxBlobAxes);
		shape_.clear();
		count_ = 1;
		shape_.resize(shape.size());
		for (int i = 0; i < shape.size(); ++i) 
		{
			CHECK_GE(shape[i], 0);
			if (count_ != 0) 
			{
				CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
			}
			count_ *= shape[i];
			shape_[i] = shape[i];
		}
		if (count_ > capacity_) 
		{
			capacity_ = count_;
			data_ = new Avalon_Synchronizer<Dtype>(count_, gpu_device_, mode_);
		}
	}

	template <typename Dtype>
	const Dtype* Pandora_Blob<Dtype>::cpu_data() const
	{
		CHECK(data_);
		return (const Dtype*)data_->cpu_data();
	}

	template <typename Dtype>
	void Pandora_Blob<Dtype>::set_cpu_data(Dtype* data) 
	{
		CHECK(data);
		data_->set_cpu_data(data);
	}

	template <typename Dtype>
	void Pandora_Blob<Dtype>::set_cpu_data(Dtype* data, size_t size)
	{
		CHECK(data);
		data_->set_cpu_data(data, size);
	}

	template <typename Dtype>
	const Dtype* Pandora_Blob<Dtype>::gpu_data() const 
	{
		CHECK(data_);
		return (const Dtype*)data_->gpu_data();
	}

	template <typename Dtype>
	void Pandora_Blob<Dtype>::set_gpu_data(Dtype* data)
	{
		CHECK(data);
		data_->set_gpu_data(data);
	}

	template <typename Dtype>
	Dtype* Pandora_Blob<Dtype>::mutable_cpu_data() 
	{
		CHECK(data_);
		return static_cast<Dtype*>(data_->mutable_cpu_data());
	}

	template <typename Dtype>
	Dtype* Pandora_Blob<Dtype>::mutable_gpu_data() 
	{
		CHECK(data_);
		return static_cast<Dtype*>(data_->mutable_gpu_data());
	}

#ifdef USE_MKLDNN
	template <typename Dtype>
	const Dtype* Pandora_Blob<Dtype>::prv_data() const
	{
		CHECK(data_);
		return (const Dtype*)data_->prv_data();
	}

	template <typename Dtype>
	Dtype* Pandora_Blob<Dtype>::mutable_prv_data()
	{
		CHECK(data_);
		return static_cast<Dtype*>(data_->mutable_prv_data());
	}

	template <typename Dtype>
	PrvMemDescr* Pandora_Blob<Dtype>::get_prv_data_descriptor()
	{
		CHECK(data_);
		return data_->prv_descriptor_;
	}

	template <typename Dtype>
	void Pandora_Blob<Dtype>::set_prv_data_descriptor(PrvMemDescr* descriptor, bool same_data)
	{
		CHECK(data_);
		data_->set_prv_descriptor(descriptor, same_data);
	}
#endif

#ifdef CAFFEMODEL_SUPPORT
	template <typename Dtype>
	void Pandora_Blob<Dtype>::FromProto(const caffe::BlobProto& proto, bool reshape) {
		if (reshape)
		{
			std::vector<int> shape;
			if (proto.has_num() || proto.has_channels() ||
				proto.has_height() || proto.has_width())
			{
				// Using deprecated 4D Blob dimensions --
				// shape is (num, channels, height, width).
				shape.resize(4);
				shape[0] = proto.num();
				shape[1] = proto.channels();
				shape[2] = proto.height();
				shape[3] = proto.width();
			}
			else 
			{
				shape.resize(proto.shape().dim_size());
				for (int i = 0; i < proto.shape().dim_size(); ++i) 
				{
					shape[i] = proto.shape().dim(i);
				}
			}
			Reshape(shape);
		}
		else {
			//CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
		}
		// copy data
		/*Dtype* data_vec = mutable_cpu_data();
		if (proto.double_data_size() > 0) {
			CHECK_EQ(count_, proto.double_data_size());
			for (int i = 0; i < count_; ++i) {
				data_vec[i] = proto.double_data(i);
			}
		}
		else {
			CHECK_EQ(count_, proto.data_size());
			for (int i = 0; i < count_; ++i) {
				data_vec[i] = proto.data(i);
			}
		}*/
		CHECK_EQ(count_, proto.data_size());
		Dtype* data_vec = new Dtype[count_];
		for (int i = 0; i < count_; ++i)
		{
			data_vec[i] = proto.data(i);
		}
		set_cpu_data(data_vec, count_*sizeof(Dtype));
		//delete data_vec;
	}

	template class Pandora_Blob<float>;
	template class Pandora_Blob<double>;
#endif
}