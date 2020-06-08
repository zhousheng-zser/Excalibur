#include "abi/param_string.hpp"
#include "abi/platform_encoding.hpp"
#include "memory.hpp"

#include <new>
#include <atomic>
#include <string>
#include <cstring>
#include <type_traits>
#include <string_view>

namespace glasssix::exposing::allocations
{
	namespace
	{
		struct param_string_header
		{
			utf8_char* data;
			std::size_t size;
			std::atomic_uint32_t reference_count;
		};

		param_string_header* create_param_string_header(std::size_t size) noexcept
		{
			auto buffer = memory::heap_alloc_elements<memory::byte_type>(sizeof(param_string_header) + size + 1);
			auto header = new (buffer) param_string_header;

			if (header)
			{
				// Since C++17, a reference to an old pointer used by placement new or an new pointer in a different type by casting must be forwarded by std::launder.
				// Otherwise, it is a UB.
				header->data = new (std::launder(buffer) + sizeof(param_string_header)) utf8_char[size + 1];
				header->size = size;
				header->reference_count = 1;

				std::memset(header->data, 0, header->size);
			}

			return header;
		}
	}

	EXPORT_EXCALIBUR_PRIMITIVES param_string_handle G6_ABI_CALL create_param_string(const utf8_char* str, std::size_t size) noexcept
	{
		if (str == nullptr)
		{
			return nullptr;
		}

		if (auto header = create_param_string_header(size))
		{
			std::memcpy(header->data, str, size);

			return pure_c::to_handle<param_string_handle>(header);
		}

		return nullptr;
	}

	EXPORT_EXCALIBUR_PRIMITIVES param_string_handle G6_ABI_CALL create_param_string_from_narrow(const char* narrow_str, std::size_t size) noexcept
	{
#ifdef _WIN32
		return narrow_str ?
			platform_encoding::win32::wide_to_multibyte(
				platform_encoding::win32::narrow_to_wide(std::string_view{ narrow_str, size }),
				true,
				[](std::size_t size) { return std::tuple{ create_param_string_header(size), [](param_string_header* header) { return header->data; } }; },
				[](param_string_header*& header) { memory::heap_free(header); header = nullptr; }
			) :
			nullptr;
#else
		if (narrow_str == nullptr)
		{
			return nullptr;
		}

		if (auto header = create_param_string_header(size))
		{
			return (std::memcpy(header->data, narrow_str, size), header);
		}

		return nullptr;
#endif
	}

	EXPORT_EXCALIBUR_PRIMITIVES param_string_handle G6_ABI_CALL create_param_string_ref(param_string_handle str) noexcept
	{
		if (str == nullptr)
		{
			return 0;
		}

		return (pure_c::from_handle<param_string_header>(str)->reference_count.fetch_add(1, std::memory_order_relaxed), str);
	}

	EXPORT_EXCALIBUR_PRIMITIVES param_string_handle G6_ABI_CALL duplicate_param_string(param_string_handle str) noexcept
	{
		if (str == nullptr)
		{
			return nullptr;
		}

		auto header = pure_c::from_handle<param_string_header>(str);

		return create_param_string(header->data, header->size);
	}

	EXPORT_EXCALIBUR_PRIMITIVES param_string_handle G6_ABI_CALL concat_param_string(param_string_handle left, param_string_handle right) noexcept
	{
		if (left == nullptr || right == nullptr)
		{
			return nullptr;
		}

		auto header_left = pure_c::from_handle<param_string_header>(left);
		auto header_right = pure_c::from_handle<param_string_header>(right);

		if (auto concat_header = create_param_string_header(header_left->size + header_right->size))
		{
			std::memcpy(concat_header->data, header_left->data, header_left->size);
			std::memcpy(concat_header->data + header_left->size, header_right->data, header_right->size);

			return pure_c::to_handle<param_string_handle>(concat_header);
		}

		return nullptr;
	}

	EXPORT_EXCALIBUR_PRIMITIVES std::uint32_t G6_ABI_CALL free_param_string(param_string_handle str) noexcept
	{
		if (str == nullptr)
		{
			return 0;
		}

		std::uint32_t count = pure_c::from_handle<param_string_header>(str)->reference_count.fetch_sub(1, std::memory_order_release) - 1;

		if (count == 0)
		{
			std::atomic_thread_fence(std::memory_order_acquire);
			memory::heap_free(str);
		}

		return count;
	}

	EXPORT_EXCALIBUR_PRIMITIVES const utf8_char* G6_ABI_CALL get_param_string_data(param_string_handle str) noexcept
	{
		return str ? pure_c::from_handle<param_string_header>(str)->data : nullptr;
	}

	EXPORT_EXCALIBUR_PRIMITIVES std::size_t G6_ABI_CALL get_param_string_size(param_string_handle str) noexcept
	{
		return str ? pure_c::from_handle<param_string_header>(str)->size : 0;
	}
}
