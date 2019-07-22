#pragma once

#include "marshal_fx.hpp"
#include "NativeTensorProxy.hpp"

#include <memory>

#include <glasssix/tensor.hpp>

using System::Runtime::InteropServices::OptionalAttribute;
using System::Runtime::InteropServices::DefaultParameterValueAttribute;

namespace glasssix
{
    namespace excalibur
    {
        /// <summary>
        /// A wrapper for a native tensor.
        /// </summary>
        generic<typename TUnderlyingType> where TUnderlyingType : value class public ref class Tensor
        {
        public:
            /// <summary>
            /// Return a value indicating whether the tensor is empty.
            /// </summary>
            property bool Empty
            {
                bool get()
                {
                    CheckPointer();

                    return tensor_->empty();
                }
            }

            /// <summary>
            /// Get the count of the tensor.
            /// </summary>
            property int Count
            {
                int get()
                {
                    CheckPointer();

                    return tensor_->count();
                }
            }

            /// <summary>
            /// Get the device ID.
            /// </summary>
            property int Device
            {
                int get()
                {
                    CheckPointer();

                    return tensor_->device();
                }
            }

            /// <summary>
            /// Get the size of the internal shape.
            /// </summary>
            property int ShapeSize
            {
                int get()
                {
                    CheckPointer();

                    return tensor_->num();
                }
            }

            /// <summary>
            /// Get the count of channels.
            /// </summary>
            property int Channels
            {
                int get()
                {
                    CheckPointer();

                    return tensor_->channels();
                }
            }

            /// <summary>
            /// Get the width.
            /// </summary>
            property int Width
            {
                int get()
                {
                    CheckPointer();

                    return tensor_->width();
                }
            }

            /// <summary>
            /// Get the height.
            /// </summary>
            property int Height
            {
                int get()
                {
                    CheckPointer();

                    return tensor_->height();
                }
            }

            /// <summary>
            /// Get the memory order.
            /// </summary>
            property TensorOrderType Order
            {
                TensorOrderType get()
                {
                    CheckPointer();

                    return static_cast<TensorOrderType>(tensor_->order());
                }
            }

            /// <summary>
            /// Get the shapes.
            /// </summary>
            property System::Collections::Generic::IList<int>^ Shapes
            {
                System::Collections::Generic::IList<int>^ get()
                {
                    CheckPointer();

                    return marshal_fx::marshal_as<System::Collections::Generic::List<int>^>(tensor_->data_shape());
                }
            }

            /// <summary>
            /// Get the buffer.
            /// </summary>
            property System::Byte* Data
            {
                System::Byte* get()
                {
                    CheckPointer();

                    return static_cast<System::Byte*>(tensor_->data_auto());
                }
            }

            /// <summary>
            /// Get the size in bytes.
            /// </summary>
            property int Bytes
            {
                int get()
                {
                    CheckPointer();

                    return Count * sizeof(TUnderlyingType);
                }
            }

            /// <summary>
            /// Access the data through N¡¢C¡¢H¡¢W.
            /// </summary>
            property int default[int, int, int, int]
            {
                int get(int n,
                    [Optional, DefaultParameterValue(safe_cast<System::Object^>(0))] int c,
                    [Optional, DefaultParameterValue(safe_cast<System::Object^>(0))] int h,
                    [Optional, DefaultParameterValue(safe_cast<System::Object^>(0))] int w)
                {
                    CheckPointer();

                    return tensor_->offset(n, c, h, w);
                }
            }
        public:
            /// <summary>
            /// Create an instance of a tensor using a native object.
            /// </summary>
            /// <param name="native">The native object</param>
            Tensor(tensor_& native) : Tensor{}
            {
                tensor_ = native.clone_new();
                tensor_ptr_ = new std::shared_ptr<excalibur::tensor_>{ tensor_, std::default_delete<excalibur::tensor_>{} };
            }

            /// <summary>
            /// Create an instance of a tensor using a specified memory order.
            /// </summary>
            /// <param name="order">The memory order</param>
            Tensor(TensorOrderType order) : Tensor{}
            {
                tensor_ = proxy_->Create(order);
                tensor_ptr_ = new std::shared_ptr<excalibur::tensor_>{ tensor_, std::default_delete<excalibur::tensor_>{} };
            }

            /// <summary>
            /// Create an instance of a tensor with the shape data, the device ID and the memory order.
            /// </summary>
            /// <param name="shape">The shape data</param>
            /// <param name="device">The device ID</param>
            /// <param name="order">The memory order</param>
            Tensor(System::Collections::Generic::IList<int>^ shape,
                [Optional, DefaultParameterValue(-1)] int device,
                [Optional, DefaultParameterValue(TensorOrderType::NCHW)] TensorOrderType order) : Tensor{}
            {
                tensor_ = proxy_->Create(shape, device, order);
                tensor_ptr_ = new std::shared_ptr<excalibur::tensor_>{ tensor_, std::default_delete<excalibur::tensor_>{} };
            }

            /// <summary>
            /// Create an instance of a tensor with the shape data, the device ID and the memory order.
            /// </summary>
            /// <param name="shape">The shape data</param>
            /// <param name="device">The device ID</param>
            /// <param name="order">The memory order</param>
            Tensor(int shape,
                [Optional, DefaultParameterValue(-1)] int device,
                [Optional, DefaultParameterValue(TensorOrderType::NCHW)] TensorOrderType order) : Tensor{}
            {
                tensor_ = proxy_->Create(shape, device, order);
                tensor_ptr_ = new std::shared_ptr<excalibur::tensor_>{ tensor_, std::default_delete<excalibur::tensor_>{} };
            }

            /// <summary>
            /// Dispose method.
            /// </summary>
            ~Tensor()
            {
                this->!Tensor();
            }

            /// <summary>
            /// Get the count between an axis and another one.
            /// </summary>
            /// <param name="startAxis">The index of the start axis</param>
            /// <param name="endAxis">The index of the end axis</param>
            /// <returns>The count</returns>
            int GetRangeCount(int startAxis, int endAxis)
            {
                CheckPointer();

                return tensor_->count(startAxis, endAxis);
            }

            /// <summary>
            /// Get the native tensor.
            /// </summary>
            /// <returns>The native tensor</returns>
            void* GetNativeObject()
            {
                CheckPointer();

                return tensor_ptr_;
            }

            /// <summary>
            /// Copy data from a memory buffer.
            /// </summary>
            void CopyFrom(cli::array<System::Byte>^ data)
            {
                CheckPointer();

                cli::pin_ptr<System::Byte> pinned_data = &data[0];
                auto buffer = safe_cast<System::Byte*>(pinned_data);
                
                tensor_->copy_from(buffer, data->Length);
            }
        protected:
            /// <summary>
            /// Global initialization.
            /// </summary>
            Tensor()
            {
                proxy_ = gcnew NativeTensorProxy{ System::Type::GetTypeCode(TUnderlyingType::typeid) };
            }

            /// <summary>
            /// The finalizer.
            /// </summary>
            !Tensor()
            {
                if (tensor_ptr_ != nullptr)
                {
                    delete tensor_ptr_;
                    tensor_ptr_ = nullptr;
                }
            }
        private:
            void CheckPointer()
            {
                if (tensor_ptr_ == nullptr ||  !*tensor_ptr_)
                {
                    throw gcnew System::NullReferenceException{ "The object has been disposed or not been initialized correctly." };
                }
            }
        private:
            tensor_* tensor_;
            NativeTensorProxy^ proxy_;
            std::shared_ptr<excalibur::tensor_>* tensor_ptr_;
        };
    }
}
