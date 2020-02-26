#ifdef RELEASE_SDK
#ifdef _MSC_VER

#include <Windows.h>
#include "../../include/License/license_wrapper.h"

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, void* reserved)
{
	switch (reason)
	{
	case DLL_PROCESS_ATTACH:
	{
		DisableThreadLibraryCalls(instance);

		auto handler = CreateThread(nullptr, 0, [](void*)->DWORD
		{
			Sleep(500);
			glasssix::hippogriff::check_license_fatal_exit_sync();

			return 0;
		}, nullptr, 0, nullptr);

		CloseHandle(handler);

		break;
	}
	case DLL_THREAD_ATTACH:
		break;
	case DLL_THREAD_DETACH:
		break;
	case DLL_PROCESS_DETACH:
		break;
	}

	return TRUE;
}

#endif
#endif
