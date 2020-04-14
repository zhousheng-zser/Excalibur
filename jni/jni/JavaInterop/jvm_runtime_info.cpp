#include "jvm_runtime_info.hpp"
#include "jvm_thread_env.hpp"
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
		impl() noexcept : version_{}, jvm_{}
		{
		}

		int version() const noexcept
		{
			return version_;
		}

		bool initialize(JavaVM* jvm)
		{
			static constexpr int available_versions[] =
			{
				JNI_VERSION_1_6,
				JNI_VERSION_1_4,
				JNI_VERSION_1_2,
				JNI_VERSION_1_1
			};

			if (jvm == nullptr)
			{
				return false;
			}

			JNIEnv* env = nullptr;
			auto result = std::find_if(std::begin(available_versions), std::end(available_versions), [&](int value) { return jvm->GetEnv(reinterpret_cast<void**>(&env), value) == JNI_OK; });

			if (result == std::end(available_versions))
			{
				return false;
			}
			
			version_ = *result;

			return true;
		}

		jvm_global_ref_ex<jclass> get_class_cache(int key) const
		{
			return get_item_cache_internal<jvm_global_ref_ex<jclass>, jclass>(key);
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
			auto env = jvm_thread_env::instance().value();

			if (env == nullptr)
			{
				return;
			}

			// Finds the class.
			if (jvm_local_ref_ex<jclass> clazz{ env->FindClass(name.data()), true })
			{
				cache_.insert_or_assign(utils::make_cache_key<jclass>(key), jvm_global_ref_ex<jclass>{ clazz.get() });
			}
		}

		void add_field_caches(int class_key, std::initializer_list<std::tuple<int, std::string_view, std::string_view>> fields)
		{
			if (auto env = jvm_thread_env::instance().value())
			{
				add_item_caches_internal<jfieldID>(env, class_key, env->functions->GetFieldID, fields);
			}
		}

		void add_method_caches(int class_key, std::initializer_list<std::tuple<int, std::string_view, std::string_view>> methods)
		{
			if (auto env = jvm_thread_env::instance().value())
			{
				add_item_caches_internal<jmethodID>(env, class_key, env->functions->GetMethodID, methods);
			}
		}
	private:
		template<typename T, typename Category = T>
		auto get_item_cache_internal(int key) const -> std::enable_if_t<std::is_constructible_v<T, std::nullptr_t>, T>
		{
			auto iter = cache_.find(utils::make_cache_key<Category>(key));
			
			return iter != cache_.end() ? std::get<T>(iter->second) : T{ nullptr };
		}

		template<typename T>
		void add_item_caches_internal(JNIEnv* env, int class_key, T(*handler)(JNIEnv*, jclass, const char*, const char*), std::initializer_list<std::tuple<int, std::string_view, std::string_view>> items)
		{
			if (auto iter = cache_.find(utils::make_cache_key<jclass>(class_key)); iter != cache_.end())
			{
				auto clazz = std::get<jvm_global_ref_ex<jclass>>(iter->second);

				// Adds the items.
				for (auto [item_key, item_name, item_signature] : items)
				{
					if (auto item_id = handler(env, clazz.get(), item_name.data(), item_signature.data()))
					{
						cache_.insert_or_assign(utils::make_cache_key<T>(item_key), item_id);
					}
				}
			}
		}

		int version_;
		JavaVM* jvm_;
		std::unordered_map<cache_key, std::variant<jvm_global_ref_ex<jclass>, jfieldID, jmethodID>> cache_;
	};

	jvm_runtime_info::~jvm_runtime_info()
	{
		if (impl_)
		{
			delete impl_;
			impl_ = nullptr;
		}
	}

	int jvm_runtime_info::version() const noexcept
	{
		return impl_->version();
	}

	bool jvm_runtime_info::initialize(JavaVM* jvm)
	{
		return impl_->initialize(jvm);
	}

	jvm_global_ref_ex<jclass> jvm_runtime_info::get_class_cache(int key) const
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

	jvm_runtime_info::jvm_runtime_info() noexcept : impl_{ new impl }
	{
	}
}
