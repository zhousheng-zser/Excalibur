#pragma once

#include <utility>
#include <type_traits>

#include <jni.h>
#include <android/log.h>

namespace glasssix::jni
{
	template<typename Traits>
	struct android_logger
	{
		template<typename... Args>
		static auto info(const char* format, Args&&... args)
		{
			return __android_log_print(ANDROID_LOG_INFO, Traits::value, format, std::forward<Args>(args)...);
		}

		template<typename... Args>
		static auto warn(const char* format, Args&&... args)
		{
			return __android_log_print(ANDROID_LOG_WARN, Traits::value, format, std::forward<Args>(args)...);
		}
	};
}

#define DEFINE_ANDROID_LOGGER(name)												\
namespace																		\
{																				\
	struct android_logger_traits_##name											\
	{																			\
		static constexpr auto value = #name;									\
	};																			\
																				\
	using logger = glasssix::jni::android_logger<android_logger_traits_##name>;	\
}																				\
