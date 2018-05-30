#include "TensorCV.hpp"

using namespace excalibur;

namespace glasssix
{
	namespace gilgamesh
	{
		void tensorcv::resize(Tensor^ src, Tensor^ %dst, int new_height,
			int new_width, InterpolationType type, int device)
		{
			if (device < 0)
			{
				dst = gcnew Tensor(src->Num, src->Channel, new_height, new_width, device);
				resize_cpu(src, dst, new_height, new_width, type);
			}
			else
			{
				// No implementation
				return;
			}
		}

		void tensorcv::rotate_cpu(Tensor^ src, Tensor^ %dst, float theta,
			int center_x, int center_y, InterpolationType type, float v, int device)
		{
			if (device < 0)
			{
				dst = gcnew Tensor(src->Num, src->Channel, src->Height, src->Width, device);
				rotate_cpu(src, dst, theta, center_x, center_y, type, v);
			}
			else
			{
				// No implementation
				return;
			}
		}

		///PRIVATE FUNCTIONS
		void tensorcv::resize_cpu(Tensor^ src, Tensor^ %dst, int new_height,
			int new_width, InterpolationType type)
		{
			switch (type)
			{
			case glasssix::gilgamesh::InterpolationType::Nearest:
				tensoroperation::resize_cpu(src->data, dst->data,
					new_height, new_width, interpolationType::Nearest);
				break;
			case glasssix::gilgamesh::InterpolationType::Bilinear:
				tensoroperation::resize_cpu(src->data,
					dst->data,
					new_height, new_width, interpolationType::Bilinear);
				break;
			case glasssix::gilgamesh::InterpolationType::Cubic:
				tensoroperation::resize_cpu(src->data, dst->data,
					new_height, new_width, interpolationType::Cubic);
				break;
			default:
				tensoroperation::resize_cpu(src->data, dst->data,
					new_height, new_width, interpolationType::Nearest);
				break;
			}
		}

		void tensorcv::rotate_cpu(Tensor^ src, Tensor^ %dst, float theta,
			int center_x, int center_y, InterpolationType type, float v)
		{
			switch (type)
			{
			case glasssix::gilgamesh::InterpolationType::Nearest:
				tensoroperation::rotate_cpu(src->data, dst->data,
					theta, center_x, center_y, interpolationType::Nearest, v);
				break;
			case glasssix::gilgamesh::InterpolationType::Bilinear:
				tensoroperation::rotate_cpu(src->data, dst->data,
					theta, center_x, center_y, interpolationType::Bilinear, v);
				break;
			case glasssix::gilgamesh::InterpolationType::Cubic:
				tensoroperation::rotate_cpu(src->data, dst->data,
					theta, center_x, center_y, interpolationType::Cubic, v);
				break;
			default:

				NOT_IMPLEMENTED;
				break;
			}
		}
	}
}