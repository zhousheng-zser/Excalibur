#include "abi/component_loader.hpp"

#include <type_traits>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <Windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#else
#error "Unspported platform."
#endif

namespace glasssix::exposing::dll
{
	EXPORT_EXCALIBUR_PRIMITIVES dll_handle G6_ABI_CALL load_library(const utf8_char* path) noexcept
	{
#ifdef _WIN32
		return path ? pure_c::to_handle<dll_handle>(LoadLibraryA(platform_encoding::utf8_to_narrow(path).c_str())) : nullptr;
#else
		return path ? pure_c::to_handle<dll_handle>(dlopen(platform_encoding::utf8_to_narrow(path).c_str()), RTLD_LAZY | RTLD_LOCAL) : nullptr;
#endif
	}

	EXPORT_EXCALIBUR_PRIMITIVES void G6_ABI_CALL free_library(dll_handle handle) noexcept
	{
		if (handle)
		{
#ifdef _WIN32
			FreeLibrary(pure_c::from_handle<std::remove_pointer_t<HMODULE>>(handle));
#else
			dlclose(pure_c::from_handle<void>(handle));
#endif
		}
	}

	EXPORT_EXCALIBUR_PRIMITIVES void* G6_ABI_CALL get_symbol_address(dll_handle handle, const utf8_char* name) noexcept
	{
		if (handle == nullptr || name == nullptr)
		{
			return nullptr;
		}

#ifdef _WIN32
		return GetProcAddress(pure_c::from_handle<std::remove_pointer_t<HMODULE>>(handle), platform_encoding::utf8_to_narrow(name).c_str());
#else
		return dlsym(pure_c::from_handle<void>(handle), platform_encoding::utf8_to_narrow(name).c_str());
#endif
	}
}
