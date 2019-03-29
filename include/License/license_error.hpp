#pragma once

#include <stdexcept>

namespace glasssix
{
	namespace hippogriff
	{
		/// <summary>
		/// Define license error codes.
		/// </summary>
		enum class license_error_code
		{
			none,
			machine_code_failure,
			file_not_exist,
			file_open_failure,
			decryption_failure,
			parse_failure,
			invalid_license,
			update_license
		};

		/// <summary>
		/// Define the license error exception.
		/// </summary>
		class license_error : public std::exception
		{
		public:
			license_error(license_error_code code) : code_{ code }
			{
			}

			inline license_error_code code() const
			{
				return code_;
			}
		private:
			license_error_code code_;
		};
	}
}
