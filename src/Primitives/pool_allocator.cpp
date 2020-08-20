#include "pool_allocator.hpp"

#include <cstdint>

namespace glasssix::memory
{
	namespace
	{
		template<typename Object>
		pool_allocator<Object> allocator;
	}

	template<typename Object>
	pool_allocator<Object>& pool_allocator_default::get()
	{
		return allocator<Object>;
	}

	template pool_allocator<char>& pool_allocator_default::get<char>();
	template pool_allocator<float>& pool_allocator_default::get<float>();
	template pool_allocator<double>& pool_allocator_default::get<double>();
	template pool_allocator<std::uint8_t>& pool_allocator_default::get<std::uint8_t>();
	template pool_allocator<std::uint16_t>& pool_allocator_default::get<std::uint16_t>();
	template pool_allocator<std::uint32_t>& pool_allocator_default::get<std::uint32_t>();
	template pool_allocator<std::int8_t>& pool_allocator_default::get<std::int8_t>();
	template pool_allocator<std::int16_t>& pool_allocator_default::get<std::int16_t>();
	template pool_allocator<std::int32_t>& pool_allocator_default::get<std::int32_t>();
}
