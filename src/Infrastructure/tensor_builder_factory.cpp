#include "tensor_builder_factory.hpp"
#include "tensor_builder_free_image_impl.hpp"
#include "tensor_converter.hpp"

namespace glasssix
{
    namespace excalibur
    {
        /// <summary>
        /// Create an instance of a tensor builder.
        /// </summary>
        /// <param name="type">The implementation type</param>
        /// <returns>The created object</returns>
        std::shared_ptr<tensor_builder> tensor_builder_factory::create(tensor_builder_implementation type)
        {
            switch (type)
            {
            case tensor_builder_implementation::free_image:
                return std::make_shared<tensor_builder_free_image_impl>();
            default:
                break;
            }

            return nullptr;
        }
    }
}
