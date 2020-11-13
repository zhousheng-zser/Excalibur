#pragma once

#include <abi/consumer.hpp>

namespace glasssix
{
	struct log;

	/// <summary>
	/// Available log levels.
	/// </summary>
	enum class log_level : std::int32_t
	{
		/// <summary>
		/// A message that helps debug the program and find bugs exactly.
		/// </summary>
		debug,

		/// <summary>
		/// A message that informs the consumer of some suggestive tips.
		/// </summary>
		info,

		/// <summary>
		/// A warning that is presented to the consumer.
		/// </summary>
		warning,
		
		/// <summary>
		/// A serious logic error occurs now and must be resolved immediately.
		/// </summary>
		error,

		/// <summary>
		/// A fatal error occurs unexpectedly and the program must be terminated.
		/// </summary>
		fatal
	};
}

namespace glasssix::exposing::impl
{
	template<> struct abi<log>
	{
		using identity_type = type_identity_interface;
		static constexpr guid id{ "BAA262FF-5AF5-4217-853E-83AD5FBEC6C8" };

		struct type : abi_unknown_object
		{
			virtual std::int32_t G6_ABI_CALL info(abi_in_t<param_string> level) noexcept = 0;
			virtual std::int32_t G6_ABI_CALL debug(abi_in_t<param_string> message) noexcept = 0;
			virtual std::int32_t G6_ABI_CALL error(abi_in_t<param_string> message) noexcept = 0;
			virtual std::int32_t G6_ABI_CALL fatal(abi_in_t<param_string> message) noexcept = 0;
			virtual std::int32_t G6_ABI_CALL warning(abi_in_t<param_string> message) noexcept = 0;
			virtual std::int32_t G6_ABI_CALL set_log_level(abi_in_t<log_level> message) noexcept = 0;
		};
	};

	template<typename Derived>
	struct interface_vtable<Derived, log> : interface_vtable_base<Derived, log>
	{
		virtual std::int32_t G6_ABI_CALL info(abi_in_t<param_string> message) noexcept override
		{
			return abi_safe_call([&] { *result = detach_abi(this->self().info(create_from_abi<param_string>(message))); });
		}

		virtual std::int32_t G6_ABI_CALL debug(abi_in_t<param_string> message) noexcept override
		{
			return abi_safe_call([&] { *result = detach_abi(this->self().debug(create_from_abi<param_string>(message))); });
		}

		virtual std::int32_t G6_ABI_CALL error(abi_in_t<param_string> message) noexcept override
		{
			return abi_safe_call([&] { *result = detach_abi(this->self().error(create_from_abi<param_string>(message))); });
		}

		virtual std::int32_t G6_ABI_CALL fatal(abi_in_t<param_string> message) noexcept override
		{
			return abi_safe_call([&] { *result = detach_abi(this->self().fatal(create_from_abi<param_string>(message))); });
		}

		virtual std::int32_t G6_ABI_CALL warning(abi_in_t<param_string> message) noexcept override
		{
			return abi_safe_call([&] { *result = detach_abi(this->self().warning(create_from_abi<param_string>(message))); });
		}

		virtual std::int32_t G6_ABI_CALL set_log_level(abi_in_t<log_level> level) noexcept override
		{
			return abi_safe_call([&] { *result = detach_abi(this->self().set_log_level(create_from_abi<log_level>(level))); });
		}
	};

	template<> struct abi_adapter<log>
	{
		template<typename Derived>
		struct type : enable_self_abi_awareness<Derived, log>
		{
			void info(const param_string& message) const
			{
				check_abi_result(this->self_abi().info(get_abi(message)));
			}

			void debug(const param_string& message) const
			{
				check_abi_result(this->self_abi().debug(get_abi(message)));
			}

			void error(const param_string& message) const
			{
				check_abi_result(this->self_abi().error(get_abi(message)));
			}

			void fatal(const param_string& message) const
			{
				check_abi_result(this->self_abi().fatal(get_abi(message)));
			}

			void warning(const param_string& message) const
			{
				check_abi_result(this->self_abi().warning(get_abi(message)));
			}

			void set_log_level(log_level level) const
			{
				check_abi_result(this->self_abi().set_log_level(get_abi(level)));
			}
		};
	};
}

namespace glasssix
{
	struct log : exposing::inherits<log>
	{
		using inherits::inherits;
	};
}

namespace glasssix
{
	
}
