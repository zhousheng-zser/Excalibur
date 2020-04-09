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
				::operator delete[](memory);
			}
		}

		EXPORT_EXCALIBUR_PRIMITIVES void heap_free(void* memory, std::size_t size)
		{
			if (memory != nullptr)
			{
				::operator delete[](memory, size);
			}
		}
	}
}
