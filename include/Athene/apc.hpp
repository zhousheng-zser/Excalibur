#pragma once

#include "legacy_callback_wrapper.hpp"

#include <functional>

#include <Windows.h>

namespace std
{
    class thread;
}

namespace glasssix
{
    /// <summary>
    /// Utilities for APC calls.
    /// </summary>
    class apc final
    {
    public:
        static void queue(HANDLE thread, const std::function<void()>& callback);
        static void queue_nop(std::thread& thread);
    };
}
