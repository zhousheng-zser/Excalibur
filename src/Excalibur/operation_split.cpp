#include "../../include/Excalibur/operation_split.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_split<Dtype>::operation_split(const operation_param& param) : operation<Dtype>(param)
		{
			
		}

		template<typename Dtype>
		void operation_split<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			for (size_t i = 0; i < tops.size(); i++)
			{
				tops[i] = bottoms[0];
			}
		}

		INSTANCE_CLASS(operation_split);
		REGISTE(operation_split);
	}
}