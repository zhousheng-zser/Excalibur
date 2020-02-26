#pragma once

#include "Singleton.hpp"

namespace glasssix
{
	public ref class LicenseObserver sealed : Singleton<LicenseObserver^>
	{
	public:
		event System::Action^ Unauthorized;
	public:
		void Request();
	private:
		LicenseObserver();
		void RaiseUnauthorized();
	private:
		System::Action^ unauthorized_wapper_;
		System::Runtime::InteropServices::GCHandle handle_;
	};
}
