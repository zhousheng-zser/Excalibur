#pragma once
#ifndef _TENSOR_HPP_
#define _TENSOR_HPP_
#include "syncedmem.hpp"
#include <vector>

namespace glasssix
{
	namespace memory
	{
		enum orderType { NCHW, NHWC };

		/// <summary>
		/// A non-generic abstaction of glasssix::memory::tensor.
		/// </summary>
		struct tensor_
		{
			virtual bool empty() const = 0;
			virtual int num() const = 0;
			virtual int channels() const = 0;
			virtual int height() const = 0;
			virtual int width() const = 0;
			virtual int count(int start_axis, int end_axis) const = 0;
			virtual int count() const = 0;
			virtual int device() const = 0;
			virtual orderType order() const = 0;
			virtual int offset(const int n, const int c = 0, const int h = 0, const int w = 0) const = 0;
			virtual std::vector<int> data_shape() const = 0;
			virtual void* cpu_data_any() const = 0;
			virtual void* gpu_data_any() const = 0;
			virtual void* data_auto() const = 0;
			virtual tensor_* clone_new() = 0;
			virtual std::shared_ptr<tensor_> clone_shared() = 0;
			virtual void copy_from(const void* data, size_t size) = 0;
		};

		/// <summary>
		/// The four dimension tensor in N, C, H, W
		/// </summary>
		template <typename Dtype>
		class tensor : public tensor_
		{
			syncedmem<Dtype>* data_;
			pool_allocator<Dtype>* allocator_;
			std::vector<int> shape_;
			int count_;
			int device_;
			orderType order_;
			// packed count inside element
			// c/1-h-w-1  h/1-w-1  w/1-1  scalar
			// c/4-h-w-4  h/4-w-4  w/4-4  sse/neon
			// c/8-h-w-8  h/8-w-8  w/8-8  avx/fp16
			int elempack_;
			// real size of the data_offset c*h*w
			size_t nstep_;

			void set_elempack();
			int get_pack_axis_size(int ori_size);
		public:
			// empty tensor
			tensor(orderType order = NCHW, pool_allocator<Dtype>* allocator = nullptr);
			// vector
			tensor(const int c, int device = -1, orderType order = NCHW, pool_allocator<Dtype>* allocator = nullptr);
			// matrix/gray image
			tensor(const int h, const int w, int device = -1, orderType order = NCHW, pool_allocator<Dtype>* allocator = nullptr);
			// external matrix/gray image
			tensor(const int h, const int w, Dtype* data, int device = -1, orderType order = NCHW, pool_allocator<Dtype>* allocator = nullptr);
			// 3-dimension tensor/multi-channel image
			tensor(const int c, const int h, const int w, int device = -1, orderType order = NCHW, pool_allocator<Dtype>* allocator = nullptr);
			// external 3-dimension tensor/multi-channel image
			tensor(const int c, const int h, const int w, Dtype* data, int device = -1, orderType order = NCHW, pool_allocator<Dtype>* allocator = nullptr);
			// 4-dimension/any dimention tensor
			tensor(const std::vector<int>& shape, int device = -1, orderType order = NCHW, pool_allocator<Dtype>* allocator = nullptr);

			tensor(const tensor& t);
			tensor& operator=(const tensor& t);
			~tensor() {};

			tensor clone() const;
			virtual bool empty() const override;

			tensor channel_tensor_ptr(int c);

		};
	}
}
#endif // !_TENSOR_HPP_
