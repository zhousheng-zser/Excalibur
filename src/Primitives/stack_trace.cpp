#include "stack_trace.hpp"

#include <mutex>
#include <memory>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <DbgHelp.h>
#elif defined(__linux__)

#else
#error "Unsupported platform."
#endif

namespace glasssix
{
	namespace
	{
		struct stack_trace_info
		{
			void** frames;
			std::size_t size;
		};

		std::mutex mutex_;
	}

#ifdef _WIN32
	EXPORT_EXCALIBUR_PRIMITIVES stack_trace_handle create_stack_trace(std::size_t size)
	{
		auto frames = new void* [size];
		std::size_t actual_size = [&]
		{
			std::scoped_lock lock{ mutex_ };

			return CaptureStackBackTrace(0, static_cast<std::uint32_t>(size), frames, nullptr);
		}();

		return pure_c::to_handle<stack_trace_handle>(new stack_trace_info{ frames, actual_size });
	}

	EXPORT_EXCALIBUR_PRIMITIVES stack_trace_frame_handle get_stack_trace_frame(stack_trace_handle handle, std::size_t index)
	{
		if (handle == nullptr)
		{
			return nullptr;
		}

		std::scoped_lock lock{ mutex_ };

		return nullptr;
	}

	EXPORT_EXCALIBUR_PRIMITIVES bool expand_stack_trace_frame(stack_trace_frame_info& info)
	{
		std::scoped_lock lock{ mutex_ };

		return false;
	}

	EXPORT_EXCALIBUR_PRIMITIVES void free_stack_trace(stack_trace_handle handle)
	{
	}

	EXPORT_EXCALIBUR_PRIMITIVES void free_stack_trace_frame_info(stack_trace_frame_info& info)
	{
	}
#else
	EXPORT_EXCALIBUR_PRIMITIVES stack_trace_handle create_stack_trace(std::size_t frames)
	{
}

	EXPORT_EXCALIBUR_PRIMITIVES stack_trace_frame_handle get_stack_trace_frame(std::size_t index)
	{
	}

	EXPORT_EXCALIBUR_PRIMITIVES bool expand_stack_trace_frame(stack_trace_frame_info& info)
	{
	}

	EXPORT_EXCALIBUR_PRIMITIVES void free_stack_trace(stack_trace_handle handle)
	{
	}

	EXPORT_EXCALIBUR_PRIMITIVES void free_stack_trace_frame_info(stack_trace_frame_info& info)
	{
	}
#endif
}
