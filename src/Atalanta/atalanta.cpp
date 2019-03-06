#include <msclr\marshal_cppstd.h>
#include "artemis.hpp"

using namespace System;
using namespace System::Runtime::InteropServices;
using namespace System::Collections::Generic;
using namespace System::IO;

namespace glasssix
{
	public ref class atalanta
	{
	public:
		atalanta() {};
		~atalanta() {};
		!atalanta() {};

		static int GetDeviceCount()
		{
			return excalibur::artemis::GetDeviceCount();
		}

		static int GetDeviceDriverVersion(int dev)
		{
			return excalibur::artemis::GetDeviceDriverVersion(dev);
		}


		static int GetDeviceRuntimeVersion(int dev)
		{
			return excalibur::artemis::GetDeviceRuntimeVersion(dev);
		}


		static int GetDeviceCapability(int dev)
		{
			return excalibur::artemis::GetDeviceCapability(dev);
		}

		static array<float>^ GetDeviceMemory(int dev)
		{
			auto temp = excalibur::artemis::GetDeviceMemory(dev);
			auto m_array = gcnew array<float>(temp.size());
			for (int i = 0; i < temp.size(); i++)
			{
				m_array[i] = temp[i];
			}
			return m_array;
		}

		static System::String^ GetDeviceName(int dev)
		{
			return msclr::interop::marshal_as<System::String^>(excalibur::artemis::GetDeviceName(dev));
		}

		static int GetDeviceCUDACoreNum(int dev)
		{
			return excalibur::artemis::GetDeviceCUDACoreNum(dev);
		}
	};
}