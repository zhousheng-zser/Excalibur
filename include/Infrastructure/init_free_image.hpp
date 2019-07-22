#pragma once

#include <FreeImage.h>

#include <glasssix/init_once.hpp>

namespace glasssix
{
    namespace excalibur
    {
        /// <summary>
        /// Initialize the enviroment of FreeImage.
        /// </summary>
        class init_free_image : public init_once<init_free_image>
        {
        public:
            virtual ~init_free_image()
            {
                FreeImage_DeInitialise();
            }

            virtual void init_environment_core() override
            {
                FreeImage_Initialise();
            }
        };
    }
}
