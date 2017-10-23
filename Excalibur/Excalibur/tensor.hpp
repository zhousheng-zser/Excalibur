#pragma once
#ifndef _TENSOR_HPP_
#define _TENSOR_HPP_
#include "syncedmem.hpp"
namespace excalibur
{
	class tensor
	{
		syncedmem* data_;
		std::vector<int> shape_;
		int count_;
		//int capacity_;
		int device_;

	public:
		tensor(const std::vector<int>& shape, int device);
		/*tensor()
		{
			data_ = nullptr;

		};*/
		~tensor();

		const float* cpu_data() const;
		const float* gpu_data() const;
		float* mutable_cpu_data() const;
		float* mutable_gpu_data() const;
		void set_cpu_data(float* data);
		void set_gpu_data(float* data);

		int num() const { return shape_[0]; }
		int channels() const { return shape_[1]; }
		int height() const { return shape_[2]; }
		int width() const { return shape_[3]; }

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