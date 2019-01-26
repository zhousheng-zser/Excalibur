#ifndef _ROMANCIA_HPP_
#define _ROMANCIA_HPP_

#include "baseNet.hpp"



namespace glasssix
{
	namespace longinus
	{
		class Romancia
		{
		public:

			Romancia() {}

			Romancia(int device);

			~Romancia();

			void Forward(const float* input_data, unsigned num);

			void Forward(const unsigned char* input_data, unsigned num);

			void getParam(std::vector<std::vector<float> > &keypointParam, unsigned num);

		private:
			BaseNet *baseNet_;
		};
	}
}

#endif // !_ROMANCIA_HPP_