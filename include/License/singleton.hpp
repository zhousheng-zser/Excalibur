#pragma once

#include <memory>

namespace glasssix
{
	namespace hippogriff
	{
		template<typename T>
		class singleton
		{
		public:
			virtual ~singleton() = default;

			static T& instance()
			{
				static std::shared_ptr<T> resource;

				if (resource == nullptr)
				{
					resource.reset(new T{});
				}

				return *resource;
			}
		protected:
			singleton() = default;
		};

	}
}
