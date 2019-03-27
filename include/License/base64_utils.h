#pragma once

#include <string>

namespace glasssix
{
	namespace hippogriff
	{
		class base64_utils final
		{
		public:
			static std::string base64_encode(const std::string& buffer);
			static std::string base64_decode(const std::string& buffer);
		};
	}
}
