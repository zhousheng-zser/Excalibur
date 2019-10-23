#ifndef _NSG_CALCULATE_ERROR_HPP_
#define _NSG_CALCULATE_ERROR_HPP_
#include <stdexcept>

namespace glasssix {
	namespace irisviel {
		class nsg_calculate_error
			: public std::exception
		{	// base of all runtime-error exceptions
		public:
			typedef std::exception _Mybase;

			explicit nsg_calculate_error(const std::string& _Message)
				: _Mybase(_Message.c_str())
			{	// construct from message string
			}

			explicit nsg_calculate_error(const char *_Message)
				: _Mybase(_Message)
			{	// construct from message string
			}

#if _HAS_EXCEPTIONS

#else /* _HAS_EXCEPTIONS */
		protected:
			virtual void _Doraise() const
			{	// perform class-specific exception handling
				_RAISE(*this);
			}
#endif /* _HAS_EXCEPTIONS */
		};
	}
}

#endif // _NSG_CALCULATE_ERROR_HPP_