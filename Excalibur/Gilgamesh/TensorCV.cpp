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
	}
}