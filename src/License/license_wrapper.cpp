#pragma once

#include "common.hpp"
#include "watchdog.hpp"
#include "license_wrapper.h"
#include "windows_com_global.hpp"

#include <mutex>
#include <string>

namespace glasssix
{
	namespace hippogriff
	{
		bool has_updated_license = false;

		/// <summary>
		/// Check if a license is valid.
		/// </summary>
		/// <param name="compoent_name">The component name</param>
		/// <param name="error_code">The error code</param>
		/// <returns>The validation result</returns>
		bool check_license_impl(const std::string& component_name, license_error_code& error_code, bool readonly)
		{
			static std::mutex lock;
			std::lock_guard<std::mutex> locker{ lock };

			error_code = license_error_code::none;

			try
			{
				windows_com_global::instance();
				license_context context{ component_name };
				context.check(readonly);

				return true;
			}
			catch (license_error& error)
			{
				error_code = error.code();

				return false;
			}
		}

        /// <summary>
        /// Check if a license is valid and terminate the process when failed.
        /// </summary>
        /// <param name="component_name">The component name</param>
        void check_license_fatal_exit(const std::string& component_name)
        {
			license_error_code error_code;

			if (!check_license_impl(component_name, error_code, has_updated_license))
			{
				common::fatal_exit();
			}

			has_updated_license = true;
        }

        /// <summary>
        /// Check if a license is valid.
        /// </summary>
        /// <param name="compoent_name">The component name</param>
        /// <returns>The validation result</returns>
        bool check_license(const std::string& component_name)
        {
            license_error_code error_code;

            return check_license(component_name, error_code);
        }

		/// <summary>
		/// Check if a license is valid.
		/// </summary>
		/// <param name="compoent_name">The component name</param>
		/// <param name="error_code">The error code</param>
		/// <returns>The validation result</returns>
		bool check_license(const std::string& component_name, license_error_code& error_code)
		{
			return check_license_impl(component_name, error_code, false);
		}

		/// <summary>
		/// Start the watchdog timer.
		/// </summary>
		void start_watchdog(const std::string& component_name)
		{
			watchdog::instance().start(component_name);
		}
	}
}
