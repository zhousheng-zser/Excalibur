#ifndef _BASENET_HPP_
#define _BASENET_HPP_

#include <vector>



namespace glasssix
{
	namespace longinus
	{
		class BaseNet
		{
		public:

			BaseNet() {}

			BaseNet(int device) {}

			virtual ~BaseNet() {}

			virtual void Forward(const float* input_data, unsigned num) = 0;

			virtual void Forward(const unsigned char* input_data, unsigned num) = 0;

			virtual void getParam(std::vector<std::vector<float> > &keypointParam, unsigned num) = 0;

			virtual std::vector<unsigned char> alignFace(const unsigned char* ori_image, int n, int channels, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int> >landmarks) = 0;

		};
	}
}

#endif // !_BASENET_HPP_