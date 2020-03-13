#include "jvm_runtime_info.hpp"
#include "jvm_global_ref.hpp"
#include "jvm_local_ref.hpp"
#include "cache_key.hpp"

#include <variant>
#include <algorithm>
#include <unordered_map>

namespace glasssix::jni
{
	class jvm_runtime_info::impl
	{
	public:
		impl() : version_{}, env_{}
		{
		}

		int version() const noexcept
		{
			return version_;
		}

		JNIEnv* env() const noexcept
		{
			return env_;
		}

		bool env(const std::function<int(void**, int)>& handler)
		{
			static constexpr int available_versions[] =
			{
				JNI_VERSION_1_6,
				JNI_VERSION_1_4,
				JNI_VERSION_1_2,
				JNI_VERSION_1_1
			};

			if (!handler)
			{
				return false;
			}

			JNIEnv* env = nullptr;
			auto result = std::find_if(std::begin(available_versions), std::end(available_versions), [&](int value) { return handler(reinterpret_cast<void**>(&env), value) == JNI_OK; });

			if (result == std::end(available_versions))
			{
				return false;
			}

			env_ = env;
			version_ = *result;

			return true;
		}

		jclass get_class_cache(int key) const
		{
			return get_item_cache_internal<jvm_global_ref, jclass>(key);
		}

		jfieldID get_field_cache(int key) const
		{
			return get_item_cache_internal<jfieldID>(key);
		}

		jmethodID get_method_cache(int key) const
		{
			return get_item_cache_internal<jmethodID>(key);
		}

		void add_class_cache(int key, std::string_view name)
		{
			if (env_ == nullptr)
			{
				return;
			}

			// Finds the class.
			jvm_local_ref clazz{ env_, env_->FindClass(name.data()), true };

			if (clazz)
			{
				cache_.emplace(utils::make_cache_key<jclass>(key), jvm_global_ref{ env_, clazz });
			}
		}

		void add_field_caches(int class_key, std::initializer_list<std::tuple<int, std::string_view, std::string_view>> fields)
		{
			add_item_caches_internal<jfieldID>(class_key, env_->functions->GetFieldID, fields);
		}

		void add_method_caches(int class_key, std::initializer_list<std::tuple<int, std::string_view, std::string_view>> methods)
		{
			add_item_caches_internal<jmethodID>(class_key, env_->functions->GetMethodID, methods);
		}
	private:
		template<typename T, typename Category = T>
		auto get_item_cache_internal(int key) const -> std::enable_if_t<std::is_constructible_v<T, std::nullptr_t>, T>
		{
			auto iter = cache_.find(utils::make_cache_key<Category>(key));

			return iter != cache_.end() ? std::get<T>(iter->second) : T{ nullptr };
		}

		template<typename T>
		void add_item_caches_internal(int class_key, T(*handler)(JNIEnv*, jclass, const char*, const char*), std::initializer_list<std::tuple<int, std::string_view, std::string_view>> items)
		{
			if (env_ == nullptr)
			{
				return;
			}

			auto iter = cache_.find(cache_key{ class_key });

			if (iter != cache_.end())
			{
				auto clazz = std::get<jvm_global_ref>(iter->second);

				// Adds the items.
				for (auto [item_key, item_name, item_signature] : items)
				{
					auto item_id = handler(env_, clazz, item_name.data(), item_signature.data());

					if (item_id != nullptr)
					{
						cache_.emplace(utils::make_cache_key<T>(item_key), item_id);
					}
				}
			}
		}

		int version_;
		JNIEnv* env_;
		std::unordered_map<cache_key, std::variant<jvm_global_ref, jfieldID, jmethodID>> cache_;
	};

	jvm_runtime_info::~jvm_runtime_info()
	{
		if (impl_ != nullptr)
		{
			delete impl_;
			impl_ = nullptr;
		}
	}

	int jvm_runtime_info::version() const noexcept
	{
		return impl_->version();
	}

	JNIEnv* jvm_runtime_info::env() const noexcept
	{
		return impl_->env();
	}

	bool jvm_runtime_info::env(const std::function<int(void**, int)>& handler)
	{
		return impl_->env(handler);
	}

	jclass jvm_runtime_info::get_class_cache(int key) const
	{
		return impl_->get_class_cache(key);
	}

	jfieldID jvm_runtime_info::get_field_cache(int key) const
	{
		return impl_->get_field_cache(key);
	}

	jmethodID jvm_runtime_info::get_method_cache(int key) const
	{
		return impl_->get_method_cache(key);
	}

	void jvm_runtime_info::add_class_cache(int key, std::string_view name)
	{
		impl_->add_class_cache(key, name);
	}

	void jvm_runtime_info::add_field_caches(int class_key, std::initializer_list<std::tuple<int, std::string_view, std::string_view>> fields)
	{
		impl_->add_field_caches(class_key, fields);
	}

	void jvm_runtime_info::add_method_caches(int class_key, std::initializer_list<std::tuple<int, std::string_view, std::string_view>> methods)
	{
		impl_->add_method_caches(class_key, methods);
	}

	jvm_runtime_info::jvm_runtime_info() : impl_{ new impl }
	{
	}
}
