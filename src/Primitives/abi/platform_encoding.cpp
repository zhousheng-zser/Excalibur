#include "abi/platform_encoding.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace glasssix::exposing::platform_encoding::api::win32
{
#ifdef _WIN32
	EXPORT_EXCALIBUR_PRIMITIVES std::size_t  G6_ABI_CALL get_narrow_to_wide_size(encoding_codepage codepage, const char* narrow_str, std::size_t size) noexcept
	{
		return MultiByteToWideChar(static_cast<std::uint32_t>(codepage), 0, narrow_str, static_cast<int>(size), nullptr, 0);
	}

	EXPORT_EXCALIBUR_PRIMITIVES std::size_t G6_ABI_CALL G6_ABI_CALL get_wide_to_narrow_size(encoding_codepage codepage, const wchar_t* wide_str, std::size_t size) noexcept
	{
		return WideCharToMultiByte(static_cast<std::uint32_t>(codepage), 0, wide_str, static_cast<int>(size), nullptr, 0, nullptr, nullptr);
	}

	EXPORT_EXCALIBUR_PRIMITIVES std::size_t G6_ABI_CALL narrow_to_wide(encoding_codepage codepage, const char* narrow_str, std::size_t narrow_size, wchar_t* wide_char, std::size_t wide_size) noexcept
	{
		return MultiByteToWideChar(static_cast<std::uint32_t>(codepage), 0, narrow_str, static_cast<int>(narrow_size), wide_char, static_cast<int>(wide_size));
	}

	EXPORT_EXCALIBUR_PRIMITIVES std::size_t G6_ABI_CALL wide_to_narrow(encoding_codepage codepage, const wchar_t* wide_str, std::size_t wide_size, char* narrow_char, std::size_t narrow_size) noexcept
	{
		return WideCharToMultiByte(static_cast<std::uint32_t>(codepage), 0, wide_str, static_cast<int>(wide_size), narrow_char, static_cast<int>(narrow_size), nullptr, nullptr);
	}
#endif
}
