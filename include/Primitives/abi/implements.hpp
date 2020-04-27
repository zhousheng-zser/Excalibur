#pragma once

#include "meta.hpp"
#include "base.hpp"
#include "base_abi.hpp"

#include <cstddef>
#include <type_traits>

namespace glasssix::exposing::impl
{
    template <typename Derived, typename Interface>
    class producer;

    /// <summary>
    /// Produces an implementation for an interface ABI.
    /// </summary>
    template <typename Derived, typename Interface, typename Enable>
    struct produce_for
    {
        static_assert(bool{}, "The interface must be derived from glasssix::exposing::unknown_object.");
    };

    /// <summary>
    /// Produces an implementation for an interfacial ABI.
    /// </summary>
    template <typename Derived, typename Interface>
    struct produce_for<Derived, Interface, std::enable_if_t<is_interface_v<Interface>>> : abi_t<Interface>
    {
        Derived& shim() noexcept
        {
            return static_cast<Derived&>(reinterpret_cast<producer<Derived, Interface>&>(*this));
        }

        virtual bool G6_ABI_CALL query_interface(const guid& id, void** object) noexcept override
        {
            return shim().query_interface(id, object);
        }

        virtual std::uint32_t G6_ABI_CALL add_ref() noexcept override
        {
            return shim().add_ref();
        }

        virtual std::uint32_t G6_ABI_CALL release() noexcept override
        {
            return shim().release();
        }
    };

    /// <summary>
    /// A reference to a procuded implementaion.
    /// </summary>
    template <typename Interface>
    struct produced_ref : Interface
    {
        produced_ref(void* ptr) noexcept : Interface{ nullptr }
        {
            *put_abi(*this) = ptr;
        }

        ~produced_ref() noexcept
        {
            detach_abi(*this);
        }

        produced_ref(const produced_ref&) = delete;
        produced_ref(produced_ref&&) = delete;
        produced_ref& operator=(const produced_ref&) = delete;
        produced_ref& operator=(produced_ref&&) = delete;
        void* operator new(std::size_t) = delete;
    };

    /// <summary>
    /// A class that contains a produced implementation for an interfacial ABI.
    /// </summary>
    template <typename Derived, typename Interface>
    class producer
    {
    public:
        friend produce_for<Derived, Interface>;

        operator produced_ref<Interface> const() const noexcept
        {
            return const_cast<produce_for<Derived, Interface>*>(&vtable_);
        }
    private:
        produce_for<Derived, Interface> vtable_;
    };

    template <typename Derived, typename T>
    struct producers_impl;

    template <typename Derived, typename... Interfaces>
    struct producers_impl<Derived, std::tuple<Interfaces...>> : producer<Derived, Interfaces>...
    {
    };

    /// <summary>
    /// A class that contains one or more procuded implementations for interfacial ABIs.
    /// </summary>
    template<typename Derived, typename... Interfaces>
    using producers = producers_impl<Derived, meta::tuple_if_t<is_interface, std::tuple<Interfaces...>>>;
}
