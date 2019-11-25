#include "LicenseObserver.hpp"
#include "../License/license_wrapper.h"

using System::Runtime::InteropServices::Marshal;
using System::Runtime::InteropServices::GCHandle;

namespace glasssix
{
	void LicenseObserver::Request()
	{
		hippogriff::check_license_fatal_exit_sync();
	}

	LicenseObserver::LicenseObserver()
	{
		unauthorized_wapper_ += gcnew System::Action(this, &LicenseObserver::RaiseUnauthorized);

		handle_ = GCHandle::Alloc(unauthorized_wapper_);
		auto handler = static_cast<hippogriff::unauthorized_handler_type>(Marshal::GetFunctionPointerForDelegate(unauthorized_wapper_).ToPointer());

		hippogriff::set_unauthorized_handler(handler);
	}

	void LicenseObserver::RaiseUnauthorized()
	{
		Unauthorized();
	}
}
