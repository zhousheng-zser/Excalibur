#pragma once

#include "NativeTensorConstructors.hpp"

using System::Runtime::InteropServices::GCHandle;

namespace glasssix
{
    namespace memory
    {
        struct tensor_;
    }

    namespace excalibur
    {
        /// <summary>
        /// A adapter for native tensors with caches.
        /// </summary>
        ref class NativeTensorProxy
        {
        public:
            /// <summary>
            /// Static initialization.
            /// </summary>
            static NativeTensorProxy()
            {
                using System::TypeCode;

                // Cache all possible constructors.
                auto values = System::Enum::GetValues(TypeCode::typeid);
                constructors_ = gcnew System::Collections::Generic::Dictionary<System::TypeCode, NativeTensorConstructors^>{};

                for each(auto item in values)
                {
                    try
                    {
                        auto code = safe_cast<TypeCode>(item);
                        auto obj = gcnew NativeTensorConstructors{ code };

                        constructors_[code] = obj;
                    }
                    catch (System::ArgumentException^)
                    {
                        continue;
                    }
                }
            }

            /// <summary>
            /// Create an instance of NativeTensorProxy.
            /// </summary>
            /// <param name="code">The type code</param>
            NativeTensorProxy(System::TypeCode code)
            {
                if (!constructors_->TryGetValue(code, assigned_constructors_))
                {
                    throw gcnew System::ArgumentException("Unsupported type.");
                }
            }

            /// <summary>
            /// Create an instance of a tensor using a specified memory order.
            /// </summary>
            /// <param name="order">The memory order</param>
            memory::tensor_* Create(TensorOrderType order)
            {
                auto handler = assigned_constructors_->Lookup(System::Reflection::MethodBase::GetCurrentMethod());
                auto result = safe_cast<System::IntPtr>(handler->DynamicInvoke(order)).ToPointer();
                
                return handler != nullptr ? static_cast<memory::tensor_*>(result) : nullptr;
            }

            /// <summary>
            /// Create an instance of a tensor with the shape data, the device ID and the memory order.
            /// </summary>
            /// <param name="shape">The shape data</param>
            /// <param name="device">The device ID</param>
            /// <param name="order">The memory order</param>
            memory::tensor_* Create(int shape, int device, TensorOrderType order)
            {
                auto handler = assigned_constructors_->Lookup(System::Reflection::MethodBase::GetCurrentMethod());
                auto result = safe_cast<System::IntPtr>(handler->DynamicInvoke(shape, device, order)).ToPointer();

                return handler != nullptr ? static_cast<memory::tensor_*>(result) : nullptr;
            }

            /// <summary>
            /// Create an instance of a tensor with the shape data, the device ID and the memory order.
            /// </summary>
            /// <param name="shape">The shape data</param>
            /// <param name="device">The device ID</param>
            /// <param name="order">The memory order</param>
            memory::tensor_* Create(System::Collections::Generic::IList<int>^ shape, int device, TensorOrderType order)
            {
                auto handler = assigned_constructors_->Lookup(System::Reflection::MethodBase::GetCurrentMethod());
                auto result = safe_cast<System::IntPtr>(GCHandle::Alloc(handler->DynamicInvoke(shape, device, order))).ToPointer();

                return handler != nullptr ? static_cast<memory::tensor_*>(result) : nullptr;
            }
        private:
            NativeTensorConstructors^ assigned_constructors_;
        private:
            static initonly System::Collections::Generic::Dictionary<System::TypeCode, NativeTensorConstructors^>^ constructors_;
        };
    }
}
