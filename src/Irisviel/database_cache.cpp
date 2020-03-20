#include "database_cache.hpp"
#include "filesystem_utils.hpp"

namespace glasssix
{
	namespace irisviel
	{
		database_cache::operator bool() const noexcept
		{
			return manager && wrapper;
		}

		void database_cache::commit() const
		{
			manager->save_changes();
			wrapper->build(true);
		}

		void database_cache::remove_disk_files() const noexcept
		{
			utils::safe_remove_file(manager->file_path());
			utils::safe_remove_file(wrapper->cache_file_path());
		}
	}
}
