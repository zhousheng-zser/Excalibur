#pragma once

#include <mutex>
#include <memory>

namespace glasssix
{
	/// <summary>
	/// A singleton pattern.
	/// </summary>
	template<typename T>
	class singleton
	{
	public:
		virtual ~singleton() = default;

		template<typename... TArgs>
		static T& instance(TArgs... args)
		{
			static std::mutex mutex;
			static std::shared_ptr<T> resource;
			std::lock_guard<std::mutex> lock{ mutex };

			if (resource == nullptr)
			{
				resource.reset(new T{ args... });
			}

			return *resource;
		}
	protected:
		singleton() = default;
	};

	/// <summary>
	/// An init-once resource initializer.
	/// </summary>
	template<typename T>
	class init_once : public singleton<T>
	{
	public:
		void invoke()
		{
			if (!has_initialized_)
			{
				init_environment_core();
				has_initialized_ = true;
			}
		}

		virtual ~init_once() = default;
	protected:
		virtual void init_environment_core() = 0;
	private:
		bool has_initialized_ = false;
	};
}