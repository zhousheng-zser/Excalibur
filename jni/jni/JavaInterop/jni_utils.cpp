#include "jni_utils.hpp"

#include <cstddef>

#include <glasssix/scope_guard.hpp>

namespace glasssix::jni::utils
{
	std::string to_string(JNIEnv* env, jstring str)
	{
		auto chars = env->GetStringUTFChars(str, nullptr);
		auto size = static_cast<std::size_t>(env->GetStringUTFLength(str));
		scope_guard guard{ [&] { env->ReleaseStringUTFChars(str, chars); } };

		return std::string{ chars, size };
	}

	jstring to_jstring(JNIEnv* env, std::string_view str)
	{
		return env->NewStringUTF(str.data());
	}
}
