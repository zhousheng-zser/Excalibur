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
        glasssix::hippogriff::check_license_fatal_exit_sync();
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
