#include "logger.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#undef ERROR
#define GETTID GetCurrentThreadId
#elif defined(__linux__)
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#define GETTID pthread_self 
#endif

namespace glasssix
{
	std::uint32_t EXPORT_EXCALIBUR_PRIMITIVES get_current_thread_id() noexcept
	{
		return GETTID();
	}
}
