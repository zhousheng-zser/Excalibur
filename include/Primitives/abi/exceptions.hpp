#pragma once

#include "param_string.hpp"

#include <cstdint>
#include <stdexcept>
#include <unordered_map>

namespace glasssix::exposing
{
	class param_string;

	/// <summary>
	/// Standard codes of the return value of an ABI function.
	/// </summary>
	struct abi_result
	{
		std::int32_t code;

		constexpr abi_result() noexcept : code{}
		{
		}

		constexpr abi_result(std::int32_t code) noexcept : code{ code }
		{
		}

		constexpr operator std::int32_t() const noexcept
		{
			return code;
		}

		constexpr bool operator==(abi_result& right) const noexcept
		{
			return code == right.code;
		}

		constexpr bool operator!=(abi_result& right) const noexcept
		{
			return code != right.code;
		}

		constexpr bool no_error() const noexcept
		{
			return code >= 0;
		}
	};

	inline constexpr abi_result error_success{ 0 };
	inline constexpr abi_result error_failure{ -1 };
	inline constexpr abi_result error_not_implemented{ -2 };
	inline constexpr abi_result error_null_pointer{ -3 };
	inline constexpr abi_result error_invalid_argument{ -4 };
	inline constexpr abi_result error_out_of_bounds{ -5 };
	inline constexpr abi_result error_no_interface{ -6 };
	inline constexpr abi_result error_invalid_operation{ -7 };
	inline constexpr abi_result error_key_not_found{ -8 };
	inline constexpr abi_result error_bad_alloc{ -9 };

	inline const std::unordered_map<std::int32_t, const param_string> predefined_error_messages
	{
		{ error_success, u8"The operation successfully completed." },
		{ error_failure, u8"The operation failed because of an unknown error." },
		{ error_not_implemented, u8"The operation was not implemented." },
		{ error_null_pointer, u8"One of the parameters was null." },
		{ error_invalid_argument, u8"One of the parameters were invalid." },
		{ error_out_of_bounds, u8"The index was out of bounds." },
		{ error_no_interface, u8"The specified interface was not found." },
		{ error_invalid_operation, u8"The operation was invalid." },
		{ error_bad_alloc, u8"The allocation reported failure." }
	};

	/// <summary>
	/// Gets a predefined error message by a ABI result code.
	/// </summary>
	/// <param name="result">The result code</param>
	/// <returns>The error message</returns>
	inline const param_string& get_predefined_error_message(abi_result result) noexcept
	{
		auto iter = predefined_error_messages.find(result);

		return iter != predefined_error_messages.end() ? iter->second : predefined_error_messages.find(error_failure)->second;
	}

	/// <summary>
	/// An ABI exception.
	/// </summary>
	class abi_error
	{
	public:
		abi_error(abi_result result) noexcept : abi_error{ result, get_predefined_error_message(result) }
		{
		}

		abi_error(abi_result result, const param_string& what) noexcept : result_{ result }, what_{ what }
		{
		}

		abi_result result() const noexcept
		{
			return result_;
		}

		utf8_string_view what() const noexcept
		{
			return what_;
		}

		std::string what_to_narrow() const noexcept
		{
			return to_narrow(what_);
		}
	private:
		abi_result result_;
		param_string what_;
	};

	struct abi_failure : abi_error
	{
		abi_failure() noexcept : abi_error{ error_failure }
		{
		}
	};

	struct abi_not_implemented : abi_error
	{
		abi_not_implemented() noexcept : abi_error{ error_not_implemented }
		{
		}
	};

	struct abi_null_pointer : abi_error
	{
		abi_null_pointer() noexcept : abi_error{ error_null_pointer }
		{
		}
	};

	struct abi_invalid_argument : abi_error
	{
		abi_invalid_argument() noexcept : abi_error{ error_invalid_argument }
		{
		}
	};

	struct abi_out_of_bounds : abi_error
	{
		abi_out_of_bounds() noexcept : abi_error{ error_out_of_bounds }
		{
		}
	};

	struct abi_no_interface : abi_error
	{
		abi_no_interface() noexcept : abi_error{ error_no_interface }
		{
		}
	};

	struct abi_invalid_operation : abi_error
	{
		abi_invalid_operation() noexcept : abi_error{ error_invalid_operation }
		{
		}
	};

	struct abi_key_not_found : abi_error
	{
		abi_key_not_found() noexcept : abi_error{ error_key_not_found }
		{
		}
	};

	struct abi_bad_alloc : abi_error
	{
		abi_bad_alloc() noexcept : abi_error{ error_bad_alloc }
		{
		}
	};

	/// <summary>
	/// Catches the current exception and converts it to an ABI result code.
	/// </summary>
	/// <returns>The ABI result code</returns>
	abi_result current_exception_to_result() noexcept
	{
		try
		{
			throw;
		}
		catch (const std::bad_alloc&)
		{
			return error_bad_alloc;
		}
		catch (const std::out_of_range&)
		{
			return error_out_of_bounds;
		}
		catch (const std::invalid_argument&)
		{
			return error_invalid_argument;
		}
		catch (const std::exception&)
		{
			return error_failure;
		}
		catch (const abi_error& e)
		{
			return e.result();
		}
	}

	/// <summary>
	/// Invokes a call and forwards all the exceptions.
	/// </summary>
	/// <typeparam name="Callable">The callable type</typeparam>
	/// <param name="handler">The handler</param>
	/// <returns>The result code</returns>
	template<typename Callable>
	abi_result abi_safe_call(Callable&& handler) noexcept try
	{
		return (std::forward<Callable>(handler)(), error_success);
	}
	catch (...)
	{
		return current_exception_to_result();
	}

	/// <summary>
	/// Checks a result code and throws an exception if neccessary.
	/// </summary>
	/// <param name="result">The result code</param>
	void check_abi_result(abi_result result)
	{
		if (!result.no_error())
		{
			switch (result)
			{
			case error_failure:
				throw abi_failure{};
			case error_not_implemented:
				throw abi_not_implemented{};
			case error_null_pointer:
				throw abi_null_pointer{};
			case error_invalid_argument:
				throw abi_invalid_argument{};
			case error_out_of_bounds:
				throw abi_out_of_bounds{};
			case error_no_interface:
				throw abi_no_interface{};
			case error_invalid_operation:
				throw abi_invalid_operation{};
			case error_key_not_found:
				throw abi_key_not_found{};
			case error_bad_alloc:
				throw abi_bad_alloc{};
			default:
				throw abi_failure{};
			}
		}
	}
}
