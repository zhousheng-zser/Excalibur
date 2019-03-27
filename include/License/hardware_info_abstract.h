#pragma once

#include <string>

namespace glasssix
{
	namespace hippogriff
	{
		/// <summary>
		/// Platform-dependent hardware info
		/// </summary>
		struct hardware_info_abstract
		{
			virtual ~hardware_info_abstract() = default;

			/// <summary>
			/// Get the machine code.
			/// </summary>
			/// <returns>The machine code</returns>
			virtual std::string machine_code() const = 0;
		};
	}
}
