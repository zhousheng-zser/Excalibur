#pragma once
#ifndef _TENSOR_HPP_
#define _TENSOR_HPP_
#include "syncedmem.hpp"
namespace excalibur
{
	template <typename Dtype>
	class tensor
	{
		std::shared_ptr<syncedmem> data_;
		std::vector<int> shape_;
		int count_;
		int device_;

	public:
		// empty
		tensor();
		// universal constructor
		tensor(const std::vector<int>& shape, int device);
		// external vector
		tensor(const int shape, int device = -1);
		// copy constructor
		tensor(const tensor& t);
		// assign
		tensor& operator=(const tensor& t);
		// release
		~tensor();
		// deep copy
		tensor clone() const;
		bool empty() const;

		tensor channel_tensor_ptr(int c);

		const Dtype* cpu_data() const;
		const Dtype* gpu_data() const;
		Dtype* mutable_cpu_data() const;
		Dtype* mutable_gpu_data() const;
		void set_cpu_data(Dtype* data);
		void set_gpu_data(Dtype* data);

		int num() const
		{
			if (shape_.size()<1)
			{
				LOG(ERROR) << "out of index.";
				return 0;
			}
			return shape_[0];
		}

		int channels() const
		{
			if (shape_.size()<2)
			{
				LOG(ERROR) << "out of index.";
				return 0;
			}
			return shape_[1];
		}

		int height() const
		{
			if (shape_.size()<3)
			{
				LOG(ERROR) << "out of index.";
				return 0;
			}
			return shape_[2];
		}

		int width() const
		{
			if (shape_.size()<4)
			{
				LOG(ERROR) << "out of index.";
				return 0;
			}
			return shape_[3];
		}

		int count(int start_axis, int end_axis) const
		{
			int count = 1;
			for (int i = start_axis; i < end_axis; ++i) {
				count *= shape_[i];
			}
			return count;
		}

		int count() const
		{
			return count(0, shape_.size());
		}

		int device() const
		{
			return device_;
		}

		int offset(const int n, const int c = 0,
			const int h = 0, const int w = 0) const {
			return ((n * channels() + c) * height() + h) * width() + w;
		}

		std::vector<int> data_shape() const
		{
			return shape_;
		}
	};
}


#endif //_TENSOR_HPP_