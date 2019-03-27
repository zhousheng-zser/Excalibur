#pragma once

#include "singleton.hpp"

#ifdef _MSC_VER
#include <comdef.h>
#endif

namespace glasssix
{
	namespace hippogriff
	{
		/// <summary>
		/// Initialize and uninitialize COM.
		/// </summary>
		class windows_com_global final : public singleton<windows_com_global>
		{
		public:
			windows_com_global()
			{
#ifdef _MSC_VER
                auto hr = CoInitializeEx(0, COINIT_MULTITHREADED);

                if (SUCCEEDED(hr) || RPC_E_CHANGED_MODE == hr)
                {
                    hr = CoInitializeSecurity(nullptr, -1, nullptr, nullptr, RPC_C_AUTHN_LEVEL_PKT, RPC_C_IMP_LEVEL_IMPERSONATE, nullptr, EOAC_NONE, nullptr);
                }

                if (FAILED(hr))
                {
                    exit(0);
                }
#endif
			}

			virtual ~windows_com_global()
			{
#ifdef _MSC_VER
				CoUninitialize();
#endif
			}
		};
	}
}
