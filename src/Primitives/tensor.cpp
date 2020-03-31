#include "../../include/Primitives/tensor.hpp"
#include "../../include/Primitives/simd_instruction_set.hpp"

namespace glasssix
{
	namespace memory
	{
		// Aligns a buffer size to the specified number of bytes
		// The function returns the minimum number that is greater or equal to sz and is divisible by n
		// sz Buffer size to align
		// n Alignment size that must be a power of two
		inline size_t alignSize(size_t sz, int n)
		{
			return (sz + n - 1) & -n;
		}

		template<typename Dtype>
		tensor<Dtype>::tensor(orderType order, pool_allocator<Dtype>* allocator) :order_(order), allocator_(allocator)
		{
			count_ = 0;
			shape_ = std::vector<int>{ 0,0,0,0 };
			step_ = 0;
			device_ = -1;
			data_ = nullptr;
			order_ = order;
			allocator_ = allocator;
		}

		template<typename Dtype>
		tensor<Dtype>::tensor(const int w, int device, orderType order, pool_allocator<Dtype>* allocator) :order_(order), device_(device), allocator_(allocator)
		{
			if (order_ = NCHW)
			{
				shape_ = std::vector<int>{ 1,1,1,w };
			}
			else
			{
				shape_ = std::vector<int>{ 1,1,w,1 };
			}
			if (device_ >= 0)
			{
#ifndef USE_CUDA
				NO_GPU;
#endif // CPU only
			}
			count_ = w;
			data_ = new syncedmem<Dtype>(alignSize(count_ * sizeof(Dtype), 4) / sizeof(Dtype), device_);
			data_->set_allocator(allocator_);
			step_ = w;
		}

		template<typename Dtype>
		tensor<Dtype>::tensor(const int h, const int w, int device, orderType order, pool_allocator<Dtype>* allocator) :order_(order), device_(device), allocator_(allocator)
		{
			if (order_ = NCHW)
			{
				shape_ = std::vector<int>{ 1,1,h,w };
				step_ = h * w;
			}
			else
			{
				shape_ = std::vector<int>{ 1,h,w,1 };
				step_ = w * 1;
			}
			if (device_ >= 0)
			{
#ifndef USE_CUDA
				NO_GPU;
#endif // CPU only
			}
			count_ = shape_[0] * shape_[1] * step_;
			data_ = new syncedmem<Dtype>(alignSize(count_ * sizeof(Dtype), 4) / sizeof(Dtype), device_);
			data_->set_allocator(allocator_);
		}

		template<typename Dtype>
		tensor<Dtype>::tensor(const int h, const int w, Dtype* data, int device, orderType order, pool_allocator<Dtype>* allocator) :order_(order), device_(device), allocator_(allocator)
		{
			if (order_ = NCHW)
			{
				shape_ = std::vector<int>{ 1,1,h,w };
				step_ = h * w;
			}
			else
			{
				shape_ = std::vector<int>{ 1,h,w,1 };
				step_ = w * 1;
			}
			if (device_ >= 0)
			{
#ifndef USE_CUDA
				NO_GPU;
#endif // CPU only
			}
			count_ = shape_[0] * shape_[1] * step_;
			data_ = new syncedmem<Dtype>(alignSize(count_ * sizeof(Dtype), 4) / sizeof(Dtype), device_);
			data_->set_allocator(allocator_);
			Dtype* cpu_data = data_->mutable_cpu_data();
			// set data
			memcpy(cpu_data, data, count_ * sizeof(Dtype));
		}

		template<typename Dtype>
		tensor<Dtype>::tensor(const int c, const int h, const int w, int device, orderType order, pool_allocator<Dtype>* allocator):order_(order), device_(device), allocator_(allocator)
		{
			if (order_ = NCHW)
			{
				shape_ = std::vector<int>{ 1,c,h,w };
				step_ = alignSize(h * w * sizeof(Dtype), 16) / sizeof(Dtype);
			}
			else
			{
				shape_ = std::vector<int>{ 1,h,w,c };
				step_ = alignSize(w * c * sizeof(Dtype), 16) / sizeof(Dtype);
			}
			if (device_ >= 0)
			{
#ifndef USE_CUDA
				NO_GPU;
#endif // CPU only
			}
			count_ = shape_[0] * shape_[1] * step_;
			data_ = new syncedmem<Dtype>(alignSize(count_ * sizeof(Dtype), 4) / sizeof(Dtype), device_);
			data_->set_allocator(allocator_);
		}

		template<typename Dtype>
		tensor<Dtype>::tensor(const int c, const int h, const int w, Dtype* data, int device, orderType order, pool_allocator<Dtype>* allocator):order_(order), device_(device), allocator_(allocator)
		{
			if (order_ = NCHW)
			{
				shape_ = std::vector<int>{ 1,c,h,w };
				step_ = alignSize(h * w * sizeof(Dtype), 16) / sizeof(Dtype);
		}
			else
			{
				shape_ = std::vector<int>{ 1,h,w,c };
				step_ = alignSize(w * c * sizeof(Dtype), 16) / sizeof(Dtype);
			}
			if (device_ >= 0)
			{
#ifndef USE_CUDA
				NO_GPU;
#endif // CPU only
			}
			count_ = shape_[0] * shape_[1] * step_;
			data_ = new syncedmem<Dtype>(alignSize(count_ * sizeof(Dtype), 4) / sizeof(Dtype), device_);
			data_->set_allocator(allocator_);
			Dtype* cpu_data = data_->mutable_cpu_data();
			// set data
			if (order_ = NCHW)
			{
				// copy data channel by channel
				for (size_t channel = 0; channel < c; channel++)
				{
					memcpy(cpu_data + channel * step_, data + channel * w * h, h * w * sizeof(Dtype));
				}
			}
			else
			{
				// copy data row by row
				for (size_t row = 0; row < h; row++)
				{
					memcpy(cpu_data + row * step_, data + row * w * c, w * c * sizeof(Dtype));
				}
			}
		}

		template<typename Dtype>
		tensor<Dtype>::tensor(const std::vector<int>& shape, int device, orderType order, pool_allocator<Dtype>* allocator) :order_(order), device_(device), allocator_(allocator)
		{
			CHECK_EQ(shape.size(), 4);
			shape_ = std::vector<int>(4);
			memcpy(shape_.data(), shape.data(), 4 * sizeof(int));
			if (device_ >= 0)
			{
#ifndef USE_CUDA
				NO_GPU;
#endif // CPU only
			}
			step_ = alignSize(shape_[2] * shape_[3] * sizeof(Dtype), 16) / sizeof(Dtype);
			count_ = shape_[0] * shape_[1] * step_;
			data_ = new syncedmem<Dtype>(alignSize(count_ * sizeof(Dtype), 4) / sizeof(Dtype), device_);
			data_->set_allocator(allocator_);
		}

		// DEPTRCATED!
		template <typename Dtype>
		tensor<Dtype>::tensor(const tensor<Dtype>& t)
		{
			count_ = t.count_;
			device_ = t.device_;
			shape_ = t.shape_;
			this->data_ = t.data_;
			order_ = t.order_;
		}

		// DEPTRCATED!
		template <typename Dtype>
		tensor<Dtype>& tensor<Dtype>::operator=(const tensor<Dtype>& t)
		{
			if (this == &t)
			{
				return *this;
			}
			count_ = t.count_;
			device_ = t.device_;
			shape_ = t.shape_;
			this->data_ = t.data_;
			order_ = t.order_;
			return *this;
		}

		template <typename Dtype>
		tensor<Dtype>::~tensor()
		{
			if (data_ != nullptr)
			{
				delete data_;
				data_ = nullptr;
			}
		}

		template <typename Dtype>
		tensor<Dtype> tensor<Dtype>::clone() const
		{
			if (empty())
			{
				return tensor();
			}
			tensor t(shape_, device_, order_, allocator_);
			if (device_ >= 0)
			{
#ifdef USE_CUDA
				cudaSetDevice(device_);
				CUDA_CHECK(cudaMemcpy(t.mutable_gpu_data(), this->gpu_data(), sizeof(Dtype) * count_, cudaMemcpyDefault));
#else
				NO_GPU;
#endif
			}
			else
			{
				memcpy(t.mutable_cpu_data(), this->cpu_data(), sizeof(Dtype) * count_);
			}
			return t;
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
		void tensor<Dtype>::copy_from(const void* data, size_t size)
		{
			// USE GPU
			if (device_ >= 0)
			{
#ifdef USE_CUDA
				cudaMemcpy(gpu_data_any(), data, size, cudaMemcpyDefault);
#else
				NO_GPU;
#endif
			}
			else
			{
				memcpy(cpu_data_any(), data, size);
			}
		}
	}
}