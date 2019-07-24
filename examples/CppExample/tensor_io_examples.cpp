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
                assert(builder->load_from(R"(E:\Êý¾Ý\…½Êº.jpg)"));

                // By default, the parameters are NHWC, -1.
                // Uncomment the lines below to set custom parameters.
                // builder->tensor_parameters(NHWC);
                // builder->tensor_parameters(NHWC, -1);

                // Floating-point tensors.
                auto rgb_float = builder->to_tensor_float(tensor_layout::rgb);
                auto rgba_float = builder->to_tensor_float(tensor_layout::rgba);
                auto grayscale_float = builder->to_tensor_float(tensor_layout::grayscale);

                // Uint8 tensors.
                auto rgb_uint8 = builder->to_tensor_uint8(tensor_layout::rgb);
                auto rgba_uint8 = builder->to_tensor_uint8(tensor_layout::rgba);
                auto grayscale_uint8 = builder->to_tensor_uint8(tensor_layout::grayscale);

                // Convert to grayscale and save it to the disk.
                if (rgba_float)
                {
                    auto test = *rgba_float/* | tensor_convert_layout_to<tensor_layout::grayscale>*/;
                    if (builder->from_tensor(test, tensor_layout::rgba))
                    {
                        builder->save_to(R"(F:\¹þ¹þ¹þ.png)");
                    }
                }

                // Change the underlying type.
                if (rgb_uint8)
                {
                    auto rgb_change = *rgb_uint8 | tensor_convert_to<float>;
                }
            }
        };
    }
}

int main()
{
    using namespace glasssix::excalibur;
    tensor_io_examples::loading_example();
}
