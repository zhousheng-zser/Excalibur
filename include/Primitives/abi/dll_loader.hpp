#pragma once

#include "dllexport.hpp"
#include "g6_attributes.hpp"
#include "pure_c_handle_utils.h"
#include "platform_encoding.hpp"

namespace glasssix::exposing::dll
{
	DEFINE_PURE_C_HANDLE(dll);

	extern "C" EXPORT_EXCALIBUR_PRIMITIVES dll_handle G6_ABI_CALL load_library(const utf8_char* path) noexcept;
	extern "C" EXPORT_EXCALIBUR_PRIMITIVES void G6_ABI_CALL free_library(dll_handle handle) noexcept;
	extern "C" EXPORT_EXCALIBUR_PRIMITIVES void* G6_ABI_CALL get_symbol_address(dll_handle handle, const utf8_char* name) noexcept;
}
