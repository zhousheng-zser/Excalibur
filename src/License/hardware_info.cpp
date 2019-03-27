#include "hardware_info.h"

#ifdef _MSC_VER
#include "windows_com_global.hpp"
#endif

namespace glasssix
{
	namespace hippogriff
	{
		hardware_info::hardware_info()
		{
#ifdef _MSC_VER
            windows_com_global::instance();

            auto hr = wmi_locator_.CoCreateInstance(__uuidof(WbemLocator));

            if (SUCCEEDED(hr))
            {
                hr = wmi_locator_->ConnectServer(_bstr_t{ LR"(ROOT\CIMV2)" }, nullptr, nullptr, nullptr, 0, nullptr, nullptr, &wmi_service_);
            }

            if (SUCCEEDED(hr))
            {
                hr = CoSetProxyBlanket(wmi_service_, RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, nullptr, RPC_C_AUTHN_LEVEL_CALL, RPC_C_IMP_LEVEL_IMPERSONATE, nullptr, EOAC_NONE);
            }
#endif
		}

		/// <summary>
		/// Get the machine code.
		/// </summary>
		/// <returns>The machine code</returns>
		std::string hardware_info::machine_code() const
		{
#ifdef _MSC_VER
            return select_core("Win32_Processor", "ProcessorId") +
                select_core("Win32_BaseBoard", "Product") +
                select_core("Win32_BaseBoard", "SerialNumber") +
                select_core("Win32_ComputerSystemProduct", "UUID");
#else
            std::string result;

            return result;
#endif
		}

		std::string hardware_info::select_core(const std::string& class_name, const std::string& class_member) const
		{
#ifdef _MSC_VER
            HRESULT hr;
            _variant_t variant;
            std::string result;
            unsigned long count_read;
            _bstr_t query{ "SELECT * FROM " };

            if (wmi_service_ == nullptr)
            {
                return result;
            }

            ATL::CComPtr<IWbemClassObject> wmi_class;
            ATL::CComPtr<IEnumWbemClassObject> wmi_enum_class;

            query += _bstr_t{ class_name.c_str() };
            hr = wmi_service_->ExecQuery(_bstr_t{ "WQL" }, query, WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY, nullptr, &wmi_enum_class);

            if (FAILED(hr))
            {
                return result;
            }

            count_read = 0;
            hr = wmi_enum_class->Next(WBEM_INFINITE, 1, &wmi_class, &count_read);

            if (FAILED(hr) || count_read <= 0)
            {
                return result;
            }

            hr = wmi_class->Get(_bstr_t{ class_member.c_str() }, 0, &variant, nullptr, nullptr);

            if (SUCCEEDED(hr) && variant.vt != VT_NULL)
            {
                result = static_cast<_bstr_t>(variant);
            }

            return result;
#else
            std::string result;

            return result;
#endif
		}
	}
}
