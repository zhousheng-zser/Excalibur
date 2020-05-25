#include "abi/param_string.hpp"
#include "memory.hpp"

#include <new>
#include <string>
#include <cstring>
#include <string_view>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace glasssix::exposing::allocations
{
	namespace
	{
		struct param_string_header
		{
			utf8_char* data;
			std::size_t size;
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

				std::memset(header->data, 0, header->size);
			}

			return header;
		}

#ifdef _WIN32
		template<typename T>
		std::wstring narrow_to_wide(std::basic_string_view<T> narrow_str, bool utf8) noexcept
		{
			std::wstring result;
			std::uint32_t code_page = utf8 ? CP_UTF8 : CP_ACP;

			if (int wide_size = MultiByteToWideChar(code_page, 0, narrow_str.data(), static_cast<int>(narrow_str.size()), nullptr, 0); wide_size > 0)
			{
				result.resize(wide_size);
				wide_size = MultiByteToWideChar(code_page, 0, narrow_str.data(), static_cast<int>(narrow_str.size()), result.data(), wide_size);
			}

			if (wide_size <= 0)
			{
				result.clear();
			}

			return result;
		}

		template<typename T>
		std::basic_string<T> wide_to_narrow(std::wstring_view wide_str, bool utf8) noexcept
		{
			std::basic_string<T> result;
			std::uint32_t code_page = utf8 ? CP_UTF8 : CP_ACP;

			if (int utf8_size = WideCharToMultiByte(code_page, 0, wide_str.data(), static_cast<int>(wide_str.size()), nullptr, 0, nullptr, nullptr); utf8_size > 0)
			{
				result.resize(utf8_size);
				WideCharToMultiByte(code_page, 0, wide_str.data(), static_cast<int>(wide_str.size()), reinterpret_cast<char*>(result.data()), utf8_size, nullptr, nullptr);
			}

			if (utf8_size <= 0)
			{
				result.clear();
			}

			return result;
		}
#endif
		/// <summary>
		/// Converts a narrow string to a UTF-8 string.
		/// </summary>
		/// <param name="narrow_str">The narrow string</param>
		/// <returns>The UTF-8 string</returns>
		utf8_string narrow_to_utf8(std::string_view narrow_str) noexcept
		{
#ifdef _WIN32
			return wide_to_narrow<utf8_char>(narrow_to_wide(narrow_str, false), true);
#else
			utf8_string result(narrow_str.size(), u8'\0');

			return (std::memcpy(result.data(), narrow_str.data(), narrow_str.size()), result);
#endif
		}

		/// <summary>
		/// Converts a UTF-8 string to a narrow string.
		/// </summary>
		/// <param name="utf8_str">The UTF-8 string</param>
		/// <returns>The narrow string</returns>
		std::string utf8_to_narrow(utf8_string_view utf8_str) noexcept
		{
#ifdef _WIN32
			return wide_to_narrow<char>(narrow_to_wide(utf8_str, true), false);
#else
			std::string result(utf8_str.size(), '\0');

			return (std::memcpy(result.data(), utf8_str.data(), utf8_str.size()), result);
#endif
		}
	}

	param_string_handle EXPORT_EXCALIBUR_PRIMITIVES G6_ABI_CALL create_param_string(const utf8_char* str, std::size_t size) noexcept
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

	param_string_handle EXPORT_EXCALIBUR_PRIMITIVES G6_ABI_CALL create_param_string_from_narrow(const char* narrow_str, std::size_t size) noexcept
	{
		auto utf8_str = narrow_to_utf8(std::string_view{ narrow_str, size });

		return create_param_string(utf8_str.c_str(), utf8_str.size());
	}

	param_string_handle EXPORT_EXCALIBUR_PRIMITIVES G6_ABI_CALL duplicate_param_string(param_string_handle str) noexcept
	{
		if (str == nullptr)
		{
			return nullptr;
		}

		auto header = pure_c::from_handle<param_string_header>(str);

		return create_param_string(header->data, header->size);
	}

	param_string_handle EXPORT_EXCALIBUR_PRIMITIVES G6_ABI_CALL concat_param_string(param_string_handle left, param_string_handle right) noexcept
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

	void EXPORT_EXCALIBUR_PRIMITIVES G6_ABI_CALL free_param_string(param_string_handle str) noexcept
	{
		memory::heap_free(str);
	}

	const utf8_char* EXPORT_EXCALIBUR_PRIMITIVES G6_ABI_CALL get_param_string_data(param_string_handle str) noexcept
	{
		return str ? pure_c::from_handle<param_string_header>(str)->data : nullptr;
	}

	std::size_t EXPORT_EXCALIBUR_PRIMITIVES G6_ABI_CALL get_param_string_size(param_string_handle str) noexcept
	{
		return str ? pure_c::from_handle<param_string_header>(str)->size : 0;
	}
}
