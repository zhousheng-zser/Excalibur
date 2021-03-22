#include "os_context.hpp"
#include "memory.hpp"

#include <algorithm>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <Windows.h>
#elif defined(__linux__)
#include <cstdlib>
#else
#error "Unsupported platform."
#endif

namespace glasssix::os_context
{
	EXPORT_EXCALIBUR_PRIMITIVES void free_environment_variable(char* buffer) noexcept
	{
#ifdef _WIN32
		if (buffer)
		{
			memory::heap_free(buffer);
		}
#endif
	}

	EXPORT_EXCALIBUR_PRIMITIVES char* get_environment_variable(const char* name) noexcept
	{
		if (name == nullptr)
		{
			return nullptr;
		}

#ifdef _WIN32
		std::shared_ptr<char> buffer{ memory::heap_alloc_elements<char>(_MAX_ENV), &free_environment_variable };
		
		if (std::size_t real_size = GetEnvironmentVariableA(name, buffer.get(), _MAX_ENV); real_size != 0)
		{
			auto final_buffer = memory::heap_alloc_elements<char>(real_size + 1);

			return (std::copy(buffer.get(), buffer.get() + real_size + 1, final_buffer), final_buffer);
		}
		
		return nullptr;
#elif defined(__linux__)
		return std::getenv(name);
#endif
	}

	EXPORT_EXCALIBUR_PRIMITIVES bool set_environment_variable(const char* name, const char* value) noexcept
	{
		if (name == nullptr || value == nullptr)
		{
			return false;
		}

#ifdef _WIN32
		return ::SetEnvironmentVariableA(name, value);
#elif defined(__linux__)
		return ::setenv(name, value, 1) == 0;
#endif
	}
}
