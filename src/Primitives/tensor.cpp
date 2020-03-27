#include "../../include/Primitives/tensor.hpp"
#include "../../include/Primitives/simd_instruction_set.hpp"

namespace glasssix
{
	namespace memory
	{
		template<typename Dtype>
		void tensor<Dtype>::set_elempack()
		{
#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX512_VERSION)//AVX
			elempack_ = 8;
#elif (SIMD_X86_INSTR_SET >= SIMD_X86_SSE_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_SSE4_2_VERSION)//SSE
			elempack_ = 4;
#elif SIMD_ARM_INSTR_SET = SIMD_ARM8_64_NEON_VERSION//ARM64
			elempack_ = 8;
#elif (SIMD_ARM_INSTR_SET = SIMD_ARM8_32_NEON_VERSION) || (SIMD_ARM_INSTR_SET = SIMD_ARM7_NEON_VERSION)//NEON
			elempack_ = 4;
#else//SCALAR
			elempack_ = 1;
#endif
		}

		template<typename Dtype>
		int tensor<Dtype>::get_pack_axis_size(int ori_size)
		{
			if (ori_size <= 0)
				return 0;
			return ori_size % elempack_ == 0 ? ori_size : ori_size - (ori_size%elempack_) + elempack_;
		}

		template<typename Dtype>
		tensor<Dtype>::tensor(orderType order, pool_allocator<Dtype>* allocator)
		{
			count_ = 0;
			shape_ = std::vector<int>{ 0 };
			nstep_ = 0;
			set_elempack();
			device_ = -1;
			data_ = nullptr;
			order_ = order;
			allocator_ = allocator;
		}

		template<typename Dtype>
		tensor<Dtype>::tensor(const int c, int device, orderType order, pool_allocator<Dtype>* allocator)
		{
			order_ = order;
			count_ = c;
			device_ = device;
			if (order_ = NCHW)
			{
				shape_ = std::vector<int>{ 1,c,1,1 };
			}
			else
			{
				shape_ = std::vector<int>{ 1,1,1,c };
			}
			if (device_ >= 0)
			{
#ifndef USE_CUDA
				NO_GPU;
#endif // CPU only
			}
			allocator_ = allocator;
			data_ = new syncedmem<Dtype>(get_pack_axis_size(c), device_);
			data_->set_allocator(allocator_);
			nstep_ = get_pack_axis_size(c);
		}

		template<typename Dtype>
		tensor<Dtype>::tensor(const int h, const int w, int device, orderType order, pool_allocator<Dtype>* allocator)
		{
			order_ = order;
			device_ = device;
			if (order_ = NCHW)
			{
				shape_ = std::vector<int>{ 1,1,h,w };
			}
			else
			{
				shape_ = std::vector<int>{ 1,h,w,1 };
			}
			allocator_ = allocator;
			data_ = new syncedmem<Dtype>(get_pack_axis_size(h) * w, device_);
			count_ =  h * w;
			data_->set_allocator(allocator_);
			nstep_ = get_pack_axis_size(h) * w;
		}

		template<typename Dtype>
		tensor<Dtype>::tensor(const int h, const int w, Dtype* data, int device, orderType order, pool_allocator<Dtype>* allocator)
		{
			order_ = order;
			device_ = device;
			if (order_ = NCHW)
			{
				shape_ = std::vector<int>{ 1,1,h,w };
			}
			else
			{
				shape_ = std::vector<int>{ 1,h,w,1 };
			}
			allocator_ = allocator;
			data_ = new syncedmem<Dtype>(get_pack_axis_size(h) * w, device_);
		}
	}
}