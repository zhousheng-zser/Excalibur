#include "tensor.hpp"
#include "simd_instruction_set.hpp"

#include <cstring>

namespace glasssix
{
	namespace memory
	{
		// Aligns a buffer size to the specified number of bytes
		// The function returns the minimum number that is greater or equal to sz and is divisible by n
		// sz Buffer size to align
		// n Alignment size that must be a power of two
		inline size_t align_size(size_t sz, int n)
		{
			//TODO: switch to channel aligned version
			//return (sz + n - 1) & -n;
			return sz;
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
			data_ = std::make_shared<syncedmem<Dtype>>(align_size(count_ * sizeof(Dtype), 4) / sizeof(Dtype), device_);
			data_->set_allocator(allocator_);
		}

		template<typename Dtype>
		tensor<Dtype>::tensor(const int w, int device, orderType order, pool_allocator<Dtype>* allocator) :order_(order), device_(device), allocator_(allocator)
		{
			if (order_ == NCHW)
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
			data_ = std::make_shared<syncedmem<Dtype>>(align_size(count_ * sizeof(Dtype), 4) / sizeof(Dtype), device_);
			data_->set_allocator(allocator_);
			step_ = w;
		}

		template<typename Dtype>
		tensor<Dtype>::tensor(const int h, const int w, int device, orderType order, pool_allocator<Dtype>* allocator) :order_(order), device_(device), allocator_(allocator)
		{
			if (order_ == NCHW)
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
			data_ = std::make_shared<syncedmem<Dtype>>(align_size(count_ * sizeof(Dtype), 4) / sizeof(Dtype), device_);
			data_->set_allocator(allocator_);
		}

		template<typename Dtype>
		tensor<Dtype>::tensor(const int h, const int w, Dtype* data, int device, orderType order, pool_allocator<Dtype>* allocator) :order_(order), device_(device), allocator_(allocator)
		{
			if (order_ == NCHW)
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
			data_ = std::make_shared< syncedmem<Dtype>>(align_size(count_ * sizeof(Dtype), 4) / sizeof(Dtype), device_);
			data_->set_allocator(allocator_);
			Dtype* cpu_data = data_->mutable_cpu_data();
			// set data
			memcpy(cpu_data, data, count_ * sizeof(Dtype));
		}

		template<typename Dtype>
		tensor<Dtype>::tensor(const int c, const int h, const int w, int device, orderType order, pool_allocator<Dtype>* allocator):order_(order), device_(device), allocator_(allocator)
		{
			if (order_ == NCHW)
			{
				shape_ = std::vector<int>{ 1,c,h,w };
				step_ = align_size(h * w * sizeof(Dtype), 16) / sizeof(Dtype);
			}
			else
			{
				shape_ = std::vector<int>{ 1,h,w,c };
				step_ = align_size(w * c * sizeof(Dtype), 16) / sizeof(Dtype);
			}
			if (device_ >= 0)
			{
#ifndef USE_CUDA
				NO_GPU;
#endif // CPU only
			}
			count_ = shape_[0] * shape_[1] * step_;
			data_ = std::make_shared<syncedmem<Dtype>>(align_size(count_ * sizeof(Dtype), 4) / sizeof(Dtype), device_);
			data_->set_allocator(allocator_);
		}

		template<typename Dtype>
		tensor<Dtype>::tensor(const int c, const int h, const int w, Dtype* data, int device, orderType order, pool_allocator<Dtype>* allocator):order_(order), device_(device), allocator_(allocator)
		{
			if (order_ == NCHW)
			{
				shape_ = std::vector<int>{ 1,c,h,w };
				step_ = align_size(h * w * sizeof(Dtype), 16) / sizeof(Dtype);
			}
			else
			{
				shape_ = std::vector<int>{ 1,h,w,c };
				step_ = align_size(w * c * sizeof(Dtype), 16) / sizeof(Dtype);
			}
			if (device_ >= 0)
			{
#ifndef USE_CUDA
				NO_GPU;
#endif // CPU only
			}
			count_ = shape_[0] * shape_[1] * step_;
			data_ = std::make_shared<syncedmem<Dtype>>(align_size(count_ * sizeof(Dtype), 4) / sizeof(Dtype), device_);
			data_->set_allocator(allocator_);
			Dtype* cpu_data = data_->mutable_cpu_data();
			// set data
			if (order_ == NCHW)
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
			// CHECK_LE(shape.size(), 4);
			shape_ = std::vector<int>(4);
			if (shape.size() == 1)
			{
				shape_ = { 1,shape[0],1,1 };
			}
			if (shape.size() == 2)
			{
				shape_ = { 1, 1, shape[0], shape[1] };
			}
			if (shape.size() == 3)
			{
				shape_ = { 1, shape[0], shape[1], shape[2] };
			}
			if (shape.size() == 4)
			{
				memcpy(shape_.data(), shape.data(), 4 * sizeof(int));
			}
			if (shape.size() >4)
			{
				shape_.resize(shape.size());
				memcpy(shape_.data(), shape.data(), shape.size() * sizeof(int) ) ;
			}
			if (device_ >= 0)
			{
#ifndef USE_CUDA
				NO_GPU;
#endif // CPU only
			}

			size_t size=1;
			for(int i=2; i<shape_.size(); i++)
			{
				size*=shape_[i];
			}
			
			step_ = align_size(size * sizeof(Dtype), 16) / sizeof(Dtype);

			count_ = shape_[0] * shape_[1] * step_;

			data_ = std::make_shared<syncedmem<Dtype>>(align_size(count_ * sizeof(Dtype), 4) / sizeof(Dtype), device_);
			data_->set_allocator(allocator_);
		}

		template <typename Dtype>
		tensor<Dtype>::tensor(const tensor<Dtype>& t)
		{
			count_ = t.count_;
			device_ = t.device_;
			shape_ = t.shape_;
			this->data_ = t.data_;
			order_ = t.order_;
			step_ = t.step_;
			allocator_ = t.allocator_;
			data_->set_allocator(allocator_);
		}

		template <typename Dtype>
		tensor<Dtype>::tensor(tensor<Dtype>&& t) noexcept
		{
			count_ = std::exchange(t.count_, 0);
			device_ = std::exchange( t.device_, 0);
			shape_ = std::move(t.shape_);
			this->data_ = std::exchange(t.data_, nullptr);
			order_ = std::exchange(t.order_, orderType{});
			step_ = std::exchange(t.step_, 0);
			allocator_ = std::exchange(t.allocator_, nullptr);
		}

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
			step_ = t.step_;
			allocator_ = t.allocator_;
			data_->set_allocator(allocator_);
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
				std::memcpy(t.mutable_cpu_data(), this->cpu_data(), sizeof(Dtype) * count_);
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
		void tensor<Dtype>::fill(Dtype v)
		{
			int size = count();
			Dtype* ptr = data_->mutable_cpu_data();
			for (int i = 0; i < size; i++)
			{
				ptr[i] = v;
			}
		}

		template <typename Dtype>
		void tensor<Dtype>::convert_order()
		{
			if (device_ > 0)
				NOT_IMPLEMENTED;
			else
			{
				if (order_ == NHWC)
				{
					int src_num = num();
					int src_height = height();
					int src_width = width();
					int src_channels = channels();
					int offset = src_height * src_width;

					memory::tensor<Dtype> dst_temp(std::vector<int>{src_num, src_channels, src_height, src_width}, device_, memory::NCHW, allocator_);
					Dtype* dst_data = dst_temp.mutable_cpu_data();
					const Dtype* src_data = cpu_data();

					for (int n = 0; n < src_num; n++)
					{
						int n_offset = n * src_channels * offset;

						for (int ch = 0; ch < src_channels; ++ch)
						{
							int channel_offset = ch * offset;

							for (int row = 0; row < src_height; ++row)
							{
								int row_offset = row * src_width;

								for (int col = 0; col < src_width; ++col)
								{
									dst_data[n_offset + channel_offset + row_offset + col] = src_data[n_offset + (row_offset + col) * src_channels + ch];
								}
							}
						}
					}

					*this = dst_temp;
				}
				else
					NOT_IMPLEMENTED;
			}
		}

		template <typename Dtype>
		void tensor<Dtype>::copy_from(const void* data, size_t size)
		{
			// USE GPU
			if (device_ >= 0)
			{
#ifdef USE_CUDA
				cudaMemcpy(mutable_gpu_data(), data, size, cudaMemcpyDefault);
#else
				NO_GPU;
#endif
			}
			else
			{
				std::memcpy(mutable_cpu_data(), data, size);
			}
		}

		template<typename Dtype>
		tensor<Dtype> tensor<Dtype>::channel(int c)
		{
			CHECK_GE(c, 0);
			if (order_ == NCHW)
			{
				CHECK_LE(c, shape_[1]);
				return tensor<Dtype>(shape_[2], shape_[3], (Dtype*)data_->cpu_data() + c * step_, device_, NCHW, allocator_);
			}
			else
			{
				NOT_IMPLEMENTED;
				std::terminate();
			}
		}

		template<typename Dtype>
		const tensor<Dtype> tensor<Dtype>::channel(int c) const
		{
			CHECK_GE(c, 0);
			if (order_ == NCHW)
			{
				CHECK_LE(c, shape_[1]);
				return tensor<Dtype>(shape_[2], shape_[3], (Dtype*)data_->cpu_data() + c * step_, device_, NCHW, allocator_);
			}
			else
			{
				NOT_IMPLEMENTED;
				std::terminate();
			}
		}

		template<typename Dtype>
		Dtype* tensor<Dtype>::row(int y)
		{
			CHECK_GE(y, 0);
			if (order_ == NCHW)
			{
				CHECK_LE(y, shape_[2]);
				return data_->mutable_cpu_data() + y * shape_[3];
			}
			else
			{
				CHECK_LE(y, shape_[1]);
				return data_->mutable_cpu_data() + y * step_;
			}
		}

		template<typename Dtype>
		const Dtype* tensor<Dtype>::row(int y) const
		{
			CHECK_GE(y, 0);
			if (order_ == NCHW)
			{
				CHECK_LE(y, shape_[2]);
				return data_->cpu_data() + y * shape_[3];
			}
			else
			{
				CHECK_LE(y, shape_[1]);
				return data_->cpu_data() + y * step_;
			}
		}

		template <typename Dtype>
		tensor<Dtype>::operator Dtype *()
		{
			CHECK(data_);
			return data_->mutable_cpu_data();
		}

		template <typename Dtype>
		tensor<Dtype>::operator const Dtype *() const
		{
			CHECK(data_);
			return data_->cpu_data();
		}

		template <typename Dtype>
		Dtype& tensor<Dtype>::operator[](size_t i)
		{
			CHECK(data_);
			return data_->mutable_cpu_data()[i];
		}

		template <typename Dtype>
		const Dtype& tensor<Dtype>::operator[](size_t i) const
		{
			CHECK(data_);
			return data_->cpu_data()[i];
		}

		// instantiate class
		template class tensor<float>;
		template class tensor<double>;
		template class tensor<int const*>;
		template class tensor<int>;
		template class tensor<unsigned char>;
		template class tensor<char>;
		template class tensor<signed char>;
		template class tensor<short>;
		template class tensor<unsigned int>;
		template class tensor<unsigned short>;
	}
}