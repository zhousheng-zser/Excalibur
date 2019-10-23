#include "apc.hpp"

#include <thread>

#include <Windows.h>

namespace glasssix
{
    void apc::queue(HANDLE thread, const std::function<void()>& callback)
    {
        if (thread != nullptr)
        {
            auto wrapper = std::make_shared<legacy_callback_wrapper<void()>>();
            wrapper->set([wrapper, callback = callback]
            {
                if (callback)
                {
                    callback();
                }
            });

            QueueUserAPC(wrapper->uintptr_t_adapter(), thread, reinterpret_cast<uintptr_t>(wrapper.get()));
        }
    }

    void apc::queue_nop(std::thread& thread)
    {
        if (thread.joinable())
        {
            QueueUserAPC([](uintptr_t) {}, thread.native_handle(), 0);
        }
    }
}
