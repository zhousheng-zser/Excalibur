#include "pool_allocator.hpp"

#include <cstdint>

namespace glasssix::memory
{
	namespace
	{
		template<typename UnderlyingType>
		pool_allocator<UnderlyingType> allocator;
	}

	template<typename UnderlyingType>
	pool_allocator<UnderlyingType>& pool_allocator_default<UnderlyingType>::get()
	{
		return allocator<UnderlyingType>;
	}

	template struct pool_allocator_default<char>;
	template struct pool_allocator_default<float>;
	template struct pool_allocator_default<double>;
	template struct pool_allocator_default<std::uint8_t>;
	template struct pool_allocator_default<std::uint16_t>;
	template struct pool_allocator_default<std::uint32_t>;
	template struct pool_allocator_default<std::int8_t>;
	template struct pool_allocator_default<std::int16_t>;
	template struct pool_allocator_default<std::int32_t>;
}
