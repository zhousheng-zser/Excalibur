#ifndef _CASSIUS_FEATURE_HPP_
#define _CASSIUS_FEATURE_HPP_


#ifdef EXPORT_CASSIUS
#undef EXPORT_CASSIUS
#ifdef _MSC_VER // For Windows
#ifdef _WINDLL // Dynamic lib
#define EXPORT_CASSIUS __declspec(dllexport)
#else // Static lib
#define EXPORT_CASSIUS
#endif // !_WINDLL
#elif defined(__linux__) // For Linux
#define EXPORT_CASSIUS
#endif
#else
#ifdef _MSC_VER
#define EXPORT_CASSIUS __declspec(dllimport)
#elif defined(__linux__)
#define EXPORT_CASSIUS
#endif
#endif

#include <string>
#include <vector>

namespace glasssix
{
	namespace cassius
	{
		/// <summary>
		/// A common component supporting feature extraction. 
		/// </summary>
		class EXPORT_CASSIUS CassiusFeature
		{
		public:
			class impl;

			/// <summary>
			/// Creates an instance with the default CPU.
			/// </summary>
			CassiusFeature();

			/// <summary>
			/// Creates an instance with a specified GPU core or the default CPU.
			/// </summary>
			/// <param name="device">The device ID; -1 for CPU or a non-negative number for a GPU core</param>
			CassiusFeature(int device);

			/// <summary>
			/// The copy constructor must be disabled in PImpl pattern.
			/// </summary>
			CassiusFeature(const CassiusFeature&) = delete;

			/// <summary>
			/// Destroys the instance.
			/// </summary>
			virtual ~CassiusFeature();

			/// <summary>
			/// The copy assignment operator must be disabled in PImpl pattern.
			/// </summary>
			CassiusFeature& operator=(const CassiusFeature&) = delete;

			/// <summary>
			/// Forwards the input data and gets the result.
			/// </summary>
			/// <param name="input_data">The input data arranged in specified order</param>
			/// <param name="num">The number of bitmaps within the input data</param>
			/// <param name="order">The order that the input data are arranged in</param>
			/// <returns>The feature vectors</returns>
			std::vector<std::vector<float>> Forward(const unsigned char* input_data, int num, int order = 0) const;

			/// <summary>
			/// Gets the version of the component.
			/// </summary>
			/// <returns>The version</returns>
			static const char* getVersion();
		private:
			impl* impl_;
		};
	}
}

#endif // !_CASSIUS_FEATURE_HPP_