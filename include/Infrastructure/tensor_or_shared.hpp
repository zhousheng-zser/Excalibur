#pragma once

#include <memory>

#include <glasssix/tensor.hpp>

namespace glasssix
{
    namespace excalibur
    {
        template<typename TUnderlyingType, bool shared>
        struct tensor_or_shared {};

        template<typename TUnderlyingType>
        struct tensor_or_shared<TUnderlyingType, false>
        {
            std::reference_wrapper<tensor<TUnderlyingType>> data;

            tensor_or_shared(const tensor<TUnderlyingType>& tensor) : data{ const_cast<excalibur::tensor<TUnderlyingType>&>(tensor) }
            {
            }

            inline auto operator->() const
            {
                return &data.get();
            }

            inline auto& access() const
            {
                return data.get();
            }
        };

        template<typename TUnderlyingType>
        struct tensor_or_shared<TUnderlyingType, true>
        {
            std::shared_ptr<tensor<TUnderlyingType>> data;

            tensor_or_shared(const std::shared_ptr<tensor<TUnderlyingType>>& tensor) : data{ tensor }
            {
            }

            inline auto operator->() const
            {
                return data.get();
            }

            inline auto& access() const
            {
                return *data;
            }
        };
    }
}
