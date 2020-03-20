#include "database_manager.hpp"
#include "database_business_wrapper.hpp"

#include <memory>
#include <type_traits>

namespace glasssix
{
	namespace irisviel
	{
		struct database_cache
		{
			std::shared_ptr<database_manager> manager;
			std::shared_ptr<database_business_wrapper> wrapper;

			template<typename T, typename U>
			database_cache(T&& manager, U&& wrapper) : manager{ std::forward<T>(manager) }, wrapper{ std::forward<U>(wrapper) }
			{
			}

			database_cache() = default;
			operator bool() const noexcept;
			void commit() const;
			void remove_disk_files() const noexcept;
		};
	}
}
