#pragma once

#include <memory>
#include <string>

namespace glasssix
{
    namespace excalibur
    {
        class tensor_builder;

        /// <summary>
        /// Define all implementations.
        /// </summary>
        enum class tensor_builder_implementation
        {
            free_image
        };

        /// <summary>
        /// A factory to create tensor builders.
        /// </summary>
        class tensor_builder_factory final
        {
        public:
            /// <summary>
            /// Create an instance of a tensor builder.
            /// </summary>
            /// <param name="type">The implementation type</param>
            /// <returns>The created object</returns>
            static std::shared_ptr<tensor_builder> create(tensor_builder_implementation type = tensor_builder_implementation::free_image);
        };
    }
}
