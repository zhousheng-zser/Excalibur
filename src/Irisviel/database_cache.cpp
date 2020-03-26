#include "database_cache.hpp"
#include "filesystem_utils.hpp"

namespace glasssix
{
	namespace irisviel
	{
		database_cache::~database_cache()
		{
			// Deletes the disk file if they are empty.
			if (manager && manager->empty())
			{
				utils::safe_remove_file(manager->file_path());

				if (wrapper)
				{
					utils::safe_remove_file(wrapper->cache_file_path());
				}
			}
		}

		database_cache::operator bool() const noexcept
		{
			return manager && wrapper;
		}

		void database_cache::commit() const
		{
			manager->save_changes();
			wrapper->build(true);
		}
	}
}
