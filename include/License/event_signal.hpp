#pragma once

#ifdef _MSC_VER

#include <memory>

#include <Windows.h>

namespace glasssix
{
	namespace hippogriff
	{
		class event_signal
		{
		public:
			event_signal(bool manual_reset = false) : handle_{ CreateEvent(nullptr, manual_reset, FALSE, nullptr), [](void* handle) { CloseHandle(handle); } }
			{
			}

			virtual ~event_signal() = default;

			inline void set() const
			{
				if (handle_)
				{
					SetEvent(handle_.get());
				}
			}

			inline void reset() const
			{
				if (handle_)
				{
					ResetEvent(handle_.get());
				}
			}

			inline void wait() const
			{
				WaitForSingleObject(handle_.get(), INFINITE);
			}

			inline auto wait_alertable() const
			{
				return WaitForSingleObjectEx(handle_.get(), INFINITE, TRUE);
			}

			inline auto wait(uint32_t milliseconds) const
			{
				return WaitForSingleObject(handle_.get(), milliseconds);
			}

			inline auto wait_alertable(uint32_t milliseconds) const
			{
				return WaitForSingleObjectEx(handle_.get(), milliseconds, TRUE);
			}
		private:
			std::shared_ptr<void> handle_;
		};
	}
}

#endif
