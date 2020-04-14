#include "jvm_thread_env.hpp"

#include <memory>

namespace glasssix::jni
{
	/// <summary>
	/// Some JNI functions may contain parameters that are void** or JNIEnv** across different JNI versions.
	/// We provides an adapter here to support auto-casting.
	/// </summary>
	struct env_adapter
	{
		JNIEnv* env;

		JNIEnv* get() const noexcept
		{
			return env;
		}

		operator JNIEnv** () noexcept
		{
			return &env;
		}

		operator void** () noexcept
		{
			return reinterpret_cast<void**>(&env);
		}
	};

	jvm_thread_env::jvm_thread_env() noexcept : jvm_{}, version_{}
	{
	}

	void jvm_thread_env::initialize(JavaVM* jvm, int version) noexcept
	{
		jvm_ = jvm;
		version_ = version;
	}

	JNIEnv* jvm_thread_env::value() const
	{
		if (jvm_ == nullptr)
		{
			return nullptr;
		}

		thread_local bool attached_in_native_code = false;

		// Creates a thread-local JNIEnv* using a trick.
		thread_local auto creator = [this]
		{
			env_adapter env{ nullptr };

			if (jvm_ == nullptr)
			{
				return env.get();
			}

			// Attaches the current thread if necessary.
			if (jvm_->GetEnv(env, version_) == JNI_EDETACHED)
			{
				return jvm_->AttachCurrentThread(env, nullptr) == JNI_OK ? (attached_in_native_code = true, env.get()) : nullptr;
			}

			return env.get();
		};

		thread_local auto freeing_handler = [this](JNIEnv*)
		{
			if (jvm_ && attached_in_native_code)
			{
				jvm_->DetachCurrentThread();
			}
		};

		thread_local std::shared_ptr<JNIEnv> result{ creator(), freeing_handler };

		return result.get();
	}
}
