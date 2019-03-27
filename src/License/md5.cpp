#include "md5.h"

#ifdef _MSC_VER
#include <Windows.h>
#endif

namespace glasssix
{
	namespace hippogriff
	{
#ifdef _MSC_VER
        md5_init_context_ptr md5_init_context = reinterpret_cast<md5_init_context_ptr>(GetProcAddress(GetModuleHandle("advapi32.dll"), "MD5Init"));
        md5_final_context_ptr md5_final_context = reinterpret_cast<md5_final_context_ptr>(GetProcAddress(GetModuleHandle("advapi32.dll"), "MD5Final"));
        md5_update_context_ptr md5_update_context = reinterpret_cast<md5_update_context_ptr>(GetProcAddress(GetModuleHandle("advapi32.dll"), "MD5Update"));
#else
        md5_init_context_ptr md5_init_context = nullptr;
        md5_final_context_ptr md5_final_context = nullptr;
        md5_update_context_ptr md5_update_context = nullptr;
#endif
	}
}
