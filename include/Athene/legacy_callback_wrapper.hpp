#pragma once

#include <cstdint>
#include <functional>

namespace glasssix
{
    template <typename TFunctor>
    class legacy_callback_wrapper {};

    /// <summary>
    /// A wrapper for legacy c-style callbacks with std::unique_ptr as a resource.
    /// </summary>
    template<typename TResult, typename... TArgs>
    class legacy_callback_wrapper<TResult(TArgs...)>
    {
    public:
        using function_type = std::function<TResult(TArgs...)>;
    public:
        legacy_callback_wrapper() = default;

        legacy_callback_wrapper(const function_type& callback) : callback_{ callback }
        {
        }

        legacy_callback_wrapper(function_type&& callback) : callback_{ std::move(callback) }
        {
        }

        virtual ~legacy_callback_wrapper() = default;

        inline void set(const function_type& callback)
        {
            callback_ = callback;
        }

        inline void set(function_type&& callback)
        {
            callback_ = std::move(callback);
        }

        inline auto void_ptr_adapter() const
        {
            return lagacy_routine;
        }

        inline auto intptr_t_adapter() const
        {
            return reinterpret_cast<TResult(__stdcall*)(intptr_t, TArgs...)>(lagacy_routine);
        }

        inline auto uintptr_t_adapter() const
        {
            return reinterpret_cast<TResult(__stdcall*)(uintptr_t, TArgs...)>(lagacy_routine);
        }
    private:
        static void __stdcall lagacy_routine(void* param, TArgs... args)
        {
            auto this_ptr = static_cast<legacy_callback_wrapper*>(param);
            if (this_ptr->callback_)
            {
                this_ptr->callback_(args...);
            }
        }
    private:
        function_type callback_;
    };
}
