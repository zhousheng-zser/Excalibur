#pragma once

#include <type_traits>

#include <jni.h>
#include <android/log.h>

namespace glasssix
{
	template<typename Traits>
	struct android_logger
	{
		template<typename... Args>
		constexpr auto info(Args&&... args)
		{
			return ((void)__android_log_print(ANDROID_LOG_INFO, Traits::name, std::forward<Args>(args)...));
		}

		template<typename... Args>
		constexpr auto warn(Args&&... args)
		{
			return ((void)__android_log_print(ANDROID_LOG_WARN, Traits::name, std::forward<Args>(args)...));
		}
	};
}
