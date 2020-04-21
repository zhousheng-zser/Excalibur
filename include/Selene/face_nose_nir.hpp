#ifndef _FACE_NOSE_NIR_HPP_
#define _FACE_NOSE_NIR_HPP_

#include "selene.hpp"
#include "Excalibur/tensor_operation_cpu.hpp"

namespace glasssix
{
	namespace longinus
	{
		class Face_nir_net;
		class Nose_nir_net;

		class Face_nose_nir :public Selene
		{		

		public:
			Face_nose_nir(int device);
			virtual ~Face_nose_nir();

			void face_nose_area(const std::shared_ptr<glasssix::memory::tensor<unsigned char>> &image_nir, std::vector<std::vector<int>> bbox, std::vector<std::vector<int>> landmarks, 
				std::vector<std::shared_ptr<glasssix::memory::tensor<unsigned char>>> &face_nir, std::shared_ptr<glasssix::memory::tensor<unsigned char>> &nose_nir);
			
			bool judge(const unsigned char* nir_color_image, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int>> landmarks, float thresh[2], float value[2], int order) override;

		private:
			std::shared_ptr<Face_nir_net> face_nir_net_;
			std::shared_ptr<Nose_nir_net> nose_nir_net_;
		};
	}
}
#endif