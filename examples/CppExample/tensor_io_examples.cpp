#include "../../include/Infrastructure/tensor_builder.hpp"
#include "../../include/Infrastructure/tensor_convertions.hpp"
#include "../../include/Infrastructure/tensor_builder_factory.hpp"

namespace glasssix
{
    namespace excalibur
    {
        class tensor_io_examples final
        {
        public:
            static void loading_example()
            {
                auto builder = tensor_builder_factory::create();
                assert(builder->load_from(R"(E:\Êý¾Ý\DSC00052.JPG)"));
                
                // By default, the parameters are NHWC, -1.
                // Uncomment the lines below to set custom parameters.
                // builder->tensor_parameters(NHWC);
                // builder->tensor_parameters(NHWC, -1);

                auto rgb_float = builder->to_tensor_float(tensor_layout::rgb);
                auto rgba_float = builder->to_tensor_float(tensor_layout::rgba);
                auto grayscale_float = builder->to_tensor_float(tensor_layout::grayscale);
                auto grayscale_3_float = builder->to_tensor_float(tensor_layout::grayscale_3);

            }
        };
    }
}

int main()
{
    using namespace glasssix::excalibur;
    tensor_io_examples::loading_example();
}