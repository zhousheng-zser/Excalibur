#include "abi/param_string.hpp"
#include "abi/platform_encoding.hpp"
#include "memory.hpp"

#include <new>
#include <string>
#include <cstring>
#include <cstddef>
#include <cstdint>
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
			atomic_ref_count ref_count;
		};

		param_string_header* create_param_string_header(std::size_t size) noexcept
		{
			auto buffer = memory::heap_alloc_elements<memory::byte_type>(sizeof(param_string_header) + size + 1);

			if (buffer == nullptr)
			{
				return nullptr;
			}

			if (auto header = new (buffer) param_string_header)
			{
				// Since C++17, a reference to an old pointer used by placement new or an new pointer in a different type by casting must be forwarded by std::launder.
				// Otherwise, it is a UB.
				header->data = new (std::launder(buffer) + sizeof(param_string_header)) utf8_char[size + 1];
				header->size = size;
				header->ref_count = 1;

				return (std::memset(header->data, 0, header->size), header);
			}

			return nullptr;
		}

		param_string_handle concat_c_string(const utf8_char* left, std::size_t left_size, const utf8_char* right, std::size_t right_size) noexcept
		{
			if (auto concat_header = create_param_string_header(left_size + right_size))
			{
				std::memcpy(concat_header->data, left, left_size);
				std::memcpy(concat_header->data + left_size, right, right_size);

				return pure_c::to_handle<param_string_handle>(concat_header);
			}

			return nullptr;
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
		return str ? (++pure_c::from_handle<param_string_header>(str)->ref_count, str) : nullptr;
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

	EXPORT_EXCALIBUR_PRIMITIVES param_string_handle G6_ABI_CALL concat_c_string_with_param_string(const utf8_char* left, param_string_handle right) noexcept
	{
		if (left == nullptr || right == nullptr)
		{
			return nullptr;
		}

		std::string_view left_view{ left };
		auto header_right = pure_c::from_handle<param_string_header>(right);

		return concat_c_string(left_view.data(), left_view.size(), header_right->data, header_right->size);
	}

	EXPORT_EXCALIBUR_PRIMITIVES param_string_handle G6_ABI_CALL concat_param_string_with_c_string(param_string_handle left, const utf8_char* right) noexcept
	{
		if (left == nullptr || right == nullptr)
		{
			return nullptr;
		}

		std::string_view right_view{ right };
		auto header_left = pure_c::from_handle<param_string_header>(left);

		return concat_c_string(header_left->data, header_left->size, right_view.data(), right_view.size());
	}

	EXPORT_EXCALIBUR_PRIMITIVES param_string_handle G6_ABI_CALL concat_param_string(param_string_handle left, param_string_handle right) noexcept
	{
		if (left == nullptr || right == nullptr)
		{
			return nullptr;
		}

		auto header_left = pure_c::from_handle<param_string_header>(left);
		auto header_right = pure_c::from_handle<param_string_header>(right);

		return concat_c_string(header_left->data, header_left->size, header_right->data, header_right->size);
	}

	EXPORT_EXCALIBUR_PRIMITIVES bool G6_ABI_CALL compare_c_string_with_param_string(const utf8_char* left, param_string_handle right) noexcept
	{
		if (left == nullptr || right == nullptr)
		{
			return left == right;
		}

		auto header_right = pure_c::from_handle<param_string_header>(right);

		return left == utf8_string_view{ header_right->data, header_right->size };
	}

	EXPORT_EXCALIBUR_PRIMITIVES bool G6_ABI_CALL compare_param_string_with_c_string(param_string_handle left, const utf8_char* right) noexcept
	{
		if (left == nullptr || right == nullptr)
		{
			return left == right;
		}

		auto header_left = pure_c::from_handle<param_string_header>(left);

		return utf8_string_view{ header_left->data, header_left->size } == right;
	}

	EXPORT_EXCALIBUR_PRIMITIVES bool G6_ABI_CALL compare_param_string(param_string_handle left, param_string_handle right) noexcept
	{
		if (left == nullptr || right == nullptr)
		{
			return left == right;
		}

		auto header_left = pure_c::from_handle<param_string_header>(left);
		auto header_right = pure_c::from_handle<param_string_header>(right);

		return utf8_string_view{ header_left->data, header_left->size } == utf8_string_view{ header_right->data, header_right->size };
	}

	EXPORT_EXCALIBUR_PRIMITIVES std::uint32_t G6_ABI_CALL free_param_string(param_string_handle str) noexcept
	{
		if (str == nullptr)
		{
			return 0;
		}

		std::uint32_t count = --pure_c::from_handle<param_string_header>(str)->ref_count;

		if (count == 0)
		{
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
