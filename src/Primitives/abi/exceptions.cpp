#include "abi/exceptions.hpp"
#include "fmt/format.h"
#include "pure_c_handle_utils.h"

namespace glasssix::exposing::allocations
{
	namespace
	{
		thread_local param_string current_exception_what;

		const std::unordered_map<std::int32_t, const param_string> predefined_error_messages
		{
			{ error_success, u8"The operation successfully completed." },
			{ error_success_false, u8"The operation successfully completed with a false value." },
			{ error_failure, u8"The operation failed because of an internal error." },
			{ error_not_implemented, u8"The operation was not implemented." },
			{ error_null_pointer, u8"One of the parameters was null." },
			{ error_invalid_argument, u8"One of the parameters were invalid." },
			{ error_out_of_bounds, u8"The index was out of bounds." },
			{ error_no_interface, u8"The specified interface was not found." },
			{ error_invalid_operation, u8"The operation was invalid." },
			{ error_bad_alloc, u8"The allocation reported failure." },
			{ error_not_initialized, u8"The object has not been initialized yet." }
		};

		/// <summary>
		/// Gets a predefined error message by a ABI result code.
		/// </summary>
		/// <param name="result">The result code</param>
		/// <returns>The error message</returns>
		const param_string& get_predefined_error_message(abi_result result) noexcept
		{
			auto iter = predefined_error_messages.find(result);

			return iter != predefined_error_messages.end() ? iter->second : predefined_error_messages.find(error_failure)->second;
		}
	}

	EXPORT_EXCALIBUR_PRIMITIVES void* G6_ABI_CALL get_current_exception_what() noexcept
	{
		return get_abi(current_exception_what);
	}

	EXPORT_EXCALIBUR_PRIMITIVES void G6_ABI_CALL clear_current_exception_what() noexcept
	{
		current_exception_what.clear();
	}

	EXPORT_EXCALIBUR_PRIMITIVES void G6_ABI_CALL set_current_exception_what(void* what_abi) noexcept
	{
		if (what_abi)
		{
			current_exception_what = take_over_abi_from_void_ptr{ what_abi };
		}
	}

	EXPORT_EXCALIBUR_PRIMITIVES param_string_handle G6_ABI_CALL create_error_message_from_abi_result(std::int32_t code, const char* optional_inner_narrow_what) noexcept
	{
		auto what = optional_inner_narrow_what ?
			fmt::format("[Exception Code: {}][Message: {}][Details: {}]", code, get_predefined_error_message(code), optional_inner_narrow_what) :
			fmt::format("[Exception Code: {}][Message: {}]", code, get_predefined_error_message(code));

		return detach_abi(to_param_string(what));
	}
}
