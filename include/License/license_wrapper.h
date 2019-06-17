#pragma once

#include "license_context.h"

namespace glasssix
{
	namespace hippogriff
	{
        /// <summary>
        /// Check if the license is valid.
        /// </summary>
        /// <param name="compoent_name">The component name</param>
        /// <returns>The validation result</returns>
        bool check_license(const std::string& component_name);

		/// <summary>
		/// Check if the license is valid.
		/// </summary>
		/// <param name="component_name">The component name</param>
		/// <param name="error_code">The error code</param>
		/// <returns>The validation result</returns>
        bool check_license(const std::string& component_name, license_error_code& error_code);

        /// <summary>
        /// Check if a license is valid and terminate the process when failed.
        /// </summary>
        /// <param name="component_name">The component name</param>
#ifdef _MSC_VER
        extern "C" __declspec(dllexport) void check_license_fatal_exit(const std::string& component_name);

		/// <summary>
		/// Start the watchdog timer.
		/// </summary>
        extern "C" __declspec(dllexport) void start_watchdog(const std::string& component_name);
#elif defined(__GNUC__)
		extern "C" void check_license_fatal_exit(const std::string& component_name);

		/// <summary>
		/// Start the watchdog timer.
		/// </summary>
		extern "C" void start_watchdog(const std::string& component_name);
#endif
	}
}
