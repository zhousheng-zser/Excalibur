#pragma once

#include <memory>
#include <string>
#include <tensor_builder.hpp>

namespace glasssix
{
    namespace excalibur
    {
        /// <summary>
        /// Some helper functions for Tensor I/O.
        /// </summary>
        class tensor_io final
        {
        public:
            /// <summary>
            /// Loads an image from the disk and converts it to a tensor.
            /// </summary>
            /// <typeparam name="TUnderlyingType">The underlying type</typeparam>
            /// <param name="path">The image path</param>
            /// <param name="layout">The tensor layout</param>
            /// <returns>The tensor if the operation was successful; otherwise nullptr.</returns>
            /// <remarks>
            /// Only float and uint8_t are acceptable.
            /// </remarks>
            template<typename TUnderlyingType>
            static std::shared_ptr<tensor<TUnderlyingType>> load_shared(const std::string& path, tensor_layout layout)
            {
                if (!builder_->load_from(path))
                {
                    return nullptr;
                }

                if constexpr (std::is_same_v<TUnderlyingType, float>)
                {
                    return to_tensor_float_core<true>(layout);
                }

                if constexpr (std::is_same_v<TUnderlyingType, uint8_t>)
                {
                    return to_tensor_uint8_core<true>(layout);
                }
            }

            /// <summary>
            /// Loads an image from the disk and converts it to a tensor.
            /// </summary>
            /// <typeparam name="TUnderlyingType">The underlying type</typeparam>
            /// <param name="path">The image path</param>
            /// <param name="layout">The tensor layout</param>
            /// <returns>The tensor if the operation was successful; otherwise std::nullopt.</returns>
            /// <remarks>
            /// Only float and uint8_t are acceptable.
            /// </remarks>
            template<typename TUnderlyingType>
            static std::optional<tensor<TUnderlyingType>> load(const std::string& path, tensor_layout layout)
            {
                if (!builder_->load_from(path))
                {
                    return nullptr;
                }

                if constexpr (std::is_same_v<TUnderlyingType, float>)
                {
                    return to_tensor_float_core<false>(layout);
                }

                if constexpr (std::is_same_v<TUnderlyingType, uint8_t>)
                {
                    return to_tensor_uint8_core<false>(layout);
                }
            }

            /// <summary>
            /// Save a tensor to the disk.
            /// </summary>
            /// <typeparam name="TUnderlyingType">The underlying type</typeparam>
            /// <param name="path">The saving path</param>
            /// <param name="data">The tensor data</param>
            /// <param name="layout">The layout</param>
            /// <returns>true if the operation was successful; otherwise false.</returns>
            template<typename TUnderlyingType>
            inline static bool save(const std::string& path, tensor<TUnderlyingType>& data, tensor_layout layout)
            {
                return builder_->from_tensor(data, layout) ? builder_->save_to(path) : false;
            }
        private:
            template<bool shared>
            static auto to_tensor_float_core(tensor_layout layout)
            {
                static auto handler = [&]
                {
                    if constexpr (shared)
                    {
                        return builder_->to_tensor_float_shared(layout);
                    }
                    else
                    {
                        return builder_->to_tensor_float(layout);
                    }
                };

                return handler();
            }

            template<bool shared>
            static auto to_tensor_uint8_core(tensor_layout layout)
            {
                static auto handler = [&]
                {
                    if constexpr (shared)
                    {
                        return builder_->to_tensor_uint8_shared(layout);
                    }
                    else
                    {
                        return builder_->to_tensor_uint8(layout);
                    }
                };

                return handler();
            }
        private:
            static thread_local std::shared_ptr<tensor_builder> builder_;
        };
    }
}
