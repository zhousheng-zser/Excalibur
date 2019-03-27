#include "base64_utils.h"

#include <cryptopp/base64.h>

namespace glasssix
{
	namespace hippogriff
	{
		std::string base64_utils::base64_encode(const std::string& buffer)
		{
			std::string result;
			CryptoPP::ArraySource{ reinterpret_cast<const uint8_t*>(buffer.c_str()), buffer.size(), true,
								   new CryptoPP::Base64Encoder{ new CryptoPP::StringSink{ result }} };

			return result;
		}

		std::string base64_utils::base64_decode(const std::string& buffer)
		{
			std::string result;
			CryptoPP::ArraySource{ reinterpret_cast<const uint8_t*>(buffer.c_str()), buffer.size(), true,
								   new CryptoPP::Base64Decoder{ new CryptoPP::StringSink{ result }} };

			return result;
		}
	}
}
