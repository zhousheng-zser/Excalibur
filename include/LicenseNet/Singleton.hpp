#pragma once

namespace glasssix
{
	generic<typename T> where T : ref class
	public ref class Singleton 
	{
	public:
		static property T Current
		{
			T get() { return value_->Value; }
		}
	private:
		static T CreateInstance()
		{
			return safe_cast<T>(System::Activator::CreateInstance(T::typeid, true));
		}
	private:
		static initonly System::Lazy<T>^ value_ = gcnew System::Lazy<T>(gcnew System::Func<T>(&Singleton::CreateInstance), true);
	};
}
