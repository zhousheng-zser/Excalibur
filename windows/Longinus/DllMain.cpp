#ifdef RELEASE_SDK
#ifdef _MSC_VER

#include <ppltasks.h>
#include <Windows.h>
#include "../../include/License/license_wrapper.h"

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, void* reserved)
{
    switch (reason)
    {
    case DLL_PROCESS_ATTACH:
    {
        DisableThreadLibraryCalls(instance);

        auto handle = CreateThread(nullptr, 0, [](void*)
        {
            constexpr char component_name[] = "Excalibur";
            glasssix::hippogriff::check_license_fatal_exit(component_name);

            return 0UL;
        }, nullptr, 0, nullptr);

        CloseHandle(handle);

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
