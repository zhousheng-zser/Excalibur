#ifdef _MSC_VER

#include "license_wrapper.h"

#include <memory>

#include <comdef.h>
#include <Windows.h>

namespace glasssix
{
	namespace hippogriff
	{
		unsigned long license_thread_id = 0;
	}
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, void* reserved)
{
	using namespace glasssix::hippogriff;

	static constexpr const char* component_name = "Excalibur";

	switch (reason)
	{
	case DLL_PROCESS_ATTACH:
	{
		DisableThreadLibraryCalls(instance);
		
		auto handle = CreateThread(nullptr, 0, [](void*)
		{
			auto hr = CoInitializeEx(0, COINIT_APARTMENTTHREADED);

			if (SUCCEEDED(hr) || RPC_E_CHANGED_MODE == hr)
			{
				hr = CoInitializeSecurity(nullptr, -1, nullptr, nullptr, RPC_C_AUTHN_LEVEL_PKT, RPC_C_IMP_LEVEL_IMPERSONATE, nullptr, EOAC_NONE, nullptr);
			}

			MSG msg;
			auto timer = SetTimer(nullptr, 0, 1000 * 60 * 10, nullptr);

			while (GetMessage(&msg, nullptr, 0, 0))
			{
				switch (msg.message)
				{
				case WM_ACTIVATE:
				case WM_TIMER:
				{
					check_license_fatal_exit(component_name);
					break;
				}
				default:
					break;
				}

				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}

			KillTimer(nullptr, timer);

			CoUninitialize();

			return 0UL;
		}, nullptr, 0, &license_thread_id);

		std::shared_ptr<void> handle_lifetime{ handle, [](void* inner) { CloseHandle(inner); } };

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
