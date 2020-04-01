#include "memory.hpp"

#include <new>
#include <exception>

namespace glasssix
{
	namespace memory
	{
		EXPORT_EXCALIBUR_PRIMITIVES void glasssix_terminate()
		{
			std::terminate();
		}

		EXPORT_EXCALIBUR_PRIMITIVES void* heap_alloc(std::size_t size)
		{
			auto buffer = ::new (std::nothrow) byte_type[size];

			// Allocation failure is a fatal error.
			// As is the principle, we just terminate the process.
			if (buffer == nullptr)
			{
				glasssix_terminate();
			}

			return buffer;
		}

		EXPORT_EXCALIBUR_PRIMITIVES void heap_free(void* memory)
		{
			if (memory != nullptr)
			{
				::delete[] memory;
			}
		}

		EXPORT_EXCALIBUR_PRIMITIVES void heap_free(void* memory, std::size_t size)
		{
			if (memory != nullptr)
			{
				::operator delete[](memory, size);
			}
		}

		/// Aligns a buffer size to the specified number of bytes\n
		/// The function returns the minimum number that is greater or equal to sz and is divisible by n\n
		/// sz Buffer size to align\n
		/// n Alignment size that must be a power of two\n
		template<typename _Tp>
		static _Tp* alignPtr(_Tp* ptr, int n = (int)sizeof(_Tp))
		{
			return (_Tp*)(((size_t)ptr + n - 1) & -n);
		}

		EXPORT_EXCALIBUR_PRIMITIVES void* aligned_heap_alloc(std::size_t size, std::size_t alignment)
		{
			// check the size of alignment is pow of 2
			if (alignment & (alignment - 1))
			{
				return nullptr;
			}
			auto udata = heap_alloc_elements<byte_type>(size + sizeof(void*) + alignment);
			if (!udata)
				return nullptr;
			unsigned char** adata = alignPtr((unsigned char**)udata + 1, alignment);
			adata[-1] = udata;
			return adata;
		}

		EXPORT_EXCALIBUR_PRIMITIVES void aligned_heap_free(void *memblock)
		{
			unsigned char* udata = ((unsigned char**)memblock)[-1];
			if (udata != nullptr)
			{
				heap_free(udata);
			}
		}
	}
}
