#pragma once

#include "common.hpp"
#include "hardware_info_abstract.h"

#include <string>

#ifdef _MSC_VER
#include <atlbase.h>
#include <comdef.h>
#include <WbemIdl.h>
#endif

namespace glasssix
{
	namespace hippogriff
	{
		class hardware_info : public hardware_info_abstract
		{
		public:
			hardware_info();
			virtual ~hardware_info() = default;

			/// <summary>
			/// Get the machine code.
			/// </summary>
			/// <returns>The machine code</returns>
			virtual std::string machine_code() const override;
		private:
			std::string select_core(const std::string& class_name, const std::string& class_member) const;
		private:
#ifdef _MSC_VER
            ATL::CComPtr<IWbemLocator> wmi_locator_;
            ATL::CComPtr<IWbemServices> wmi_service_;
#endif
		};
	}
}
