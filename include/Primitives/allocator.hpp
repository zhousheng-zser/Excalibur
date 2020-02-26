#pragma once

#include "memory.hpp"

#include <type_traits>

namespace glasssix
{
	namespace memory
	{
		/// <summary>
		/// A glasssix uniform memory allocator.
		/// </summary>
		template<typename Object>
		class allocator
		{
		public:
			using pointer_type = std::add_pointer_t<Object>;

			/// <summary>
			/// Allocates a piece of memory which contains elements with the type of the allocator.
			/// </summary>
			/// <param name="size">The size in elements</param>
			/// <returns>The pointer at the first element</returns>
			pointer_type allocate(std::size_t size)
			{
				return heap_alloc_elements<Object>(size);
			}

			/// <summary>
			/// Deallocates a piece of memory which contains elements with the type of the allocator.
			/// </summary>
			/// <param name="ptr">The pointer at the first element</param>
			/// <param name="size">The size in elements</param>
			void deallocate(pointer_type ptr, std::size_t size)
			{
				heap_free(ptr, size);
			}

			/// <summary>
			/// Constructs an object with the arguments of its constructor.
			/// </summary>
			/// <typeparam name="Individual">The object type</typeparam>
			/// <typeparam name="...Args">The types of the arguments of its constructor</typeparam>
			/// <param name="ptr">The object</param>
			/// <param name="...args">The arguments of its constructor</param>
			template<typename Individual, typename... Args>
			auto construct(Individual* ptr, Args&&... args) -> std::enable_if_t<std::is_constructible_v<Individual, Args...>>
			{
				if (ptr == nullptr)
				{
					glasssix_terminate();
				}

				::new (ptr) Individual{ std::forward<Args>(args)... };
			}

			/// <summary>
			/// Destroys an object
			/// </summary>
			/// <typeparam name="Individual">The object type</typeparam>
			/// <param name="ptr">The object</param>
			template<typename Individual>
			void destroy(Individual* ptr)
			{
				if (ptr != nullptr)
				{
					ptr->~Individual();
				}
			}
		};
	}
}
