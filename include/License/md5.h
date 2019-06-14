#pragma once

#include <cstdint>

namespace glasssix
{
	namespace hippogriff
	{
		struct md5_digest_context
		{
			/// <summary>
			/// The size in bits of the original data.
			/// </summary>
			uint32_t count[2];

			/// <summary>
			/// Four 32-bit decimals to contain the final digest.
			/// If the length of the message is larger than 512 bits, it's also used for intermediate results of each group of 512 bits.
			/// </summary>
			uint32_t state[4];

			/// <summary>
			/// The input buffer.
			/// </summary>
			uint8_t buffer[64];

			/// <summary>
			/// The final digest.
			/// </summary>
			uint8_t digest[16];
		};
#ifdef _MSC_VER
		using md5_init_context_ptr = void(__stdcall*)(md5_digest_context* context);
		using md5_final_context_ptr = void(__stdcall*)(md5_digest_context* context);
		using md5_update_context_ptr = void(__stdcall*)(md5_digest_context* context, const uint8_t* data, uint32_t size);
#elif defined(__GNUC__)
		using md5_init_context_ptr = void(*)(md5_digest_context* context);
		using md5_final_context_ptr = void(*)(md5_digest_context* context);
		using md5_update_context_ptr = void(*)(md5_digest_context* context, const uint8_t* data, uint32_t size);
#endif

		extern md5_init_context_ptr md5_init_context;
		extern md5_final_context_ptr md5_final_context;
		extern md5_update_context_ptr md5_update_context;
	}
}
