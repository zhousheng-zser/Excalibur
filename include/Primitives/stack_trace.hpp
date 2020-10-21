#pragma once

#include "dllexport.hpp"
#include "pure_c_handle_utils.h"

#include <cstddef>
#include <cstdint>

namespace glasssix
{
	DEFINE_PURE_C_HANDLE(stack_trace);
	DEFINE_PURE_C_HANDLE(stack_trace_frame);

	struct stack_trace_frame_info
	{
		int line;
		char* symbol;
		char* file_name;
		std::uintptr_t address;
	};

	extern "C" EXPORT_EXCALIBUR_PRIMITIVES stack_trace_handle create_stack_trace(std::size_t size);
	extern "C" EXPORT_EXCALIBUR_PRIMITIVES stack_trace_frame_handle get_stack_trace_frame(stack_trace_handle handle, std::size_t index);
	extern "C" EXPORT_EXCALIBUR_PRIMITIVES bool expand_stack_trace_frame(stack_trace_frame_info & info);
	extern "C" EXPORT_EXCALIBUR_PRIMITIVES void free_stack_trace(stack_trace_handle handle);
	extern "C" EXPORT_EXCALIBUR_PRIMITIVES void free_stack_trace_frame_info(stack_trace_frame_info & info);
}
