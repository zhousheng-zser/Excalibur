#pragma once

#include "base.hpp"
#include "base_abi.hpp"
#include "implements.hpp"
#include "exceptions.hpp"
#include "param_string.hpp"
#include "param_vector.hpp"

#include <cstddef>
#include <cstdint>

namespace glasssix::exposing
{
	template<typename T>
	struct box_value;
}

namespace glasssix::exposing::impl
{
	template<typename T>
	struct abi<box_value<T>>
	{
		using identity_type = type_identity_interface;
		static constexpr guid id{ "CEAEA735-BA42-4B48-96B3-C2F9BAA4F5E2" };

		struct type : abi_unknown_object
		{
			virtual std::int32_t get(abi_out_t<T> value) noexcept = 0;
			virtual std::int32_t set(abi_in_t<T> value) noexcept = 0;
		};
	};

	template<typename Derived, typename T>
	struct interface_vtable<Derived, box_value<T>> : interface_vtable_base<Derived, box_value>
	{
		virtual std::int32_t get(abi_out_t<T> value) noexcept override
		{
			return abi_safe_call([&] { *value = detach_abi(this->self().get()); });
		}

		virtual std::int32_t set(abi_in_t<T> value) noexcept override
		{
			return abi_safe_call([&] { this->self().set(get_abi(value)); });
		}
	};

	template<typename T>
	struct abi_adapter<box_value<T>>
	{
		template<typename Derived>
		struct type : enable_self_abi_awareness<Derived, box_value<T>>
		{
			T get() const
			{
				T result{};

				return (check_abi_result(this->self_abi().get(put_abi(result))), result);
			}

			void set(const T& value) const
			{
				check_abi_result(this->self_abi().set(get_abi(value)));
			}
		};
	};
}

namespace glasssix::exposing
{
	template<typename T>
	struct box_value : inherits<box_value<T>>
	{
		using inherits::inherits;
	};
}
