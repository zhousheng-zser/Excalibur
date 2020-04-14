#include "jni_utils.hpp"

#include <cstddef>

#include <glasssix/scope_guard.hpp>

namespace glasssix::jni::utils
{
	std::string to_string(jstring str)
	{
		if (auto env = jvm_thread_env::instance().value())
		{
			auto chars = env->GetStringUTFChars(str, nullptr);
			auto size = static_cast<std::size_t>(env->GetStringUTFLength(str));
			scope_guard guard{ [&] { env->ReleaseStringUTFChars(str, chars); } };

			return std::string(chars, size);
		}

		return std::string{};
	}

	jstring to_jstring(std::string_view str)
	{
		auto env = jvm_thread_env::instance().value();

		return env ? env->NewStringUTF(str.data()) : nullptr;
	}
}
