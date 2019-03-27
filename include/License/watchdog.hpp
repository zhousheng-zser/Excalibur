#pragma once

#include "common.hpp"
#include "singleton.hpp"
#include "license_context.h"

#ifdef _MSC_VER
#include "event_signal.hpp"
#endif

#include <mutex>
#include <atomic>
#include <memory>
#include <string>
#include <thread>

#ifdef _MSC_VER
#include <Windows.h>
#endif

namespace glasssix
{
	namespace hippogriff
	{
		/// <summary>
		/// A watchdog timer apply to the whole program.
		/// </summary>
		class watchdog : public singleton<watchdog>
		{
		public:
			virtual ~watchdog()
			{
				signal_.set();
                timer_thread_.join();
			}

			void start(const std::string& component_name)
			{
				{
					std::lock_guard<std::mutex> lock{ mutex_ };
					context_ = std::make_shared<license_context>(component_name);
				}

				if (!has_started_)
				{
#ifdef _MSC_VER
                    timer_thread_ = std::move(std::thread{ std::bind(&watchdog::timer_routine, this) });
                    has_started_ = true;
#endif
				}
			}
		private:
#ifdef _MSC_VER
            void timer_routine()
            {
                LARGE_INTEGER due_time = {};
                std::shared_ptr<void> handle{ CreateWaitableTimer(nullptr, FALSE, nullptr), [](void* inner) { CloseHandle(inner); } };

                // Set the interval to the preseted value.
                SetWaitableTimer(handle.get(), &due_time, period_, nullptr, nullptr, FALSE);

                // Wait every timer to signal until the event signals.
                while (signal_.wait_alertable() == WAIT_IO_COMPLETION)
                {
                    try
                    {
                        std::lock_guard<std::mutex> lock{ mutex_ };
                        context_->check();
                    }
                    catch (license_error&)
                    {
                        common::fatal_exit();
                    }
                }
            }
#endif
		private:
            std::mutex mutex_;
			event_signal signal_;
            std::thread timer_thread_;
			std::atomic_bool has_started_;
			std::shared_ptr<license_context> context_;
		private:
			static constexpr int period_ = 30 * 60 * 1000;
		};
	}
}
