#pragma once

#include <type_traits>

#include <jni.h>
#include <android/log.h>

template<typename Traits>
struct logger
{
	template<typename... Args>
	constexpr auto info(Args&&... args)
	{
		return ((void)__android_log_print(ANDROID_LOG_INFO, "SharedObject1", std::forward<Args>(args)...));
	}

	template<typename... Args>
	constexpr auto warn(Args&&... args)
	{
		return ((void)__android_log_print(ANDROID_LOG_WARN, "SharedObject1", std::forward<Args>(args)...));
	}
};
