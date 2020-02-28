#pragma once

#include "dllexport.hpp"

#include <cstdint>
#include <cstddef>

namespace glasssix
{
	namespace memory
	{
#if _HAS_STD_BYTE
		using byte_type = std::byte;
#else
		using byte_type = std::uint8_t;
#endif

		/// <summary>
		/// Terminates the process immediately.
		/// </summary>
		/// <remarks>
		/// Throwing exceptions across DLL boundaries is very dangerous for possible different C++ standard libraries.
		/// Thus, we simply terminate the process if any fatal error occurs.
		/// </remarks>
		[[noreturn]] EXPORT_EXCALIBUR_PRIMITIVES void glasssix_terminate();

		/// <summary>
		/// Allocates a piece of memory on the heap.
		/// </summary>
		/// <param name="size">The size in bytes</param>
		/// <returns>The memory pointer</returns>
		EXPORT_EXCALIBUR_PRIMITIVES void* heap_alloc(std::size_t size);

		/// <summary>
		/// Frees a piece of memory on the heap.
		/// </summary>
		/// <param name="memory">The memory pointer</param>
		/// <returns>The memory pointer</returns>
		EXPORT_EXCALIBUR_PRIMITIVES void heap_free(void* memory);

		/// <summary>
		/// Frees a piece of memory on the heap.
		/// </summary>
		/// <param name="memory">The memory pointer</param>
		/// <param name="size">The size in bytes</param>
		/// <returns>The memory pointer</returns>
		EXPORT_EXCALIBUR_PRIMITIVES void heap_free(void* memory, std::size_t size);

		/// <summary>
		/// Allocates a piece of memory which contains elements of the specified type on the heap.
		/// </summary>
		/// <typeparam name="Element">The element type</typeparam>
		/// <param name="size">The size of elements</param>
		/// <returns>The memory pointer</returns>
		template<typename Element>
		Element* heap_alloc_elements(std::size_t size)
		{
			return static_cast<Element*>(heap_alloc(size * sizeof(Element)));
		}
	}
}
