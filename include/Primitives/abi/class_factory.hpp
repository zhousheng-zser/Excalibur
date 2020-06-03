#pragma once

#include "base.hpp"
#include "base_abi.hpp"
#include "implements.hpp"
#include "exceptions.hpp"
#include "param_string.hpp"

#include <cstddef>
#include <cstdint>

namespace glasssix::exposing
{
	struct class_factory;
}

namespace glasssix::exposing::impl
{
	template<> struct abi<class_factory>
	{
		using identity_type = type_identity_interface;
		static constexpr guid id{ "DCE95478-E317-43C2-B5E2-42DB0ECD4BD5" };

		struct type : abi_unknown_object
		{
			virtual std::int32_t create_instance(guid id, abi_out_t<unknown_object> object) noexcept = 0;
		};
	};

	template<typename Derived>
	struct interface_vtable<Derived, class_factory> : interface_vtable_base<Derived, class_factory>
	{
		virtual std::int32_t create_instance(guid id, abi_out_t<unknown_object> object) noexcept override
		{
			return abi_safe_call([&] { *object = detach_abi(this->self().create_instance(id)); });
		}
	};

	template<> struct abi_adapter<class_factory>
	{
		template<typename Derived>
		struct type : enable_self_abi_awareness<Derived, class_factory>
		{
			unknown_object create_instance(const guid& id)
			{
				unknown_object result{ nullptr };
				
				return (check_abi_result(this->self_abi().create_instance(id, put_abi(result))), result);
			}
		};
	};
}

namespace glasssix::exposing
{
	struct class_factory : inherits<class_factory>
	{
		class_factory(std::nullptr_t = nullptr) noexcept
		{
		}
	};
}
