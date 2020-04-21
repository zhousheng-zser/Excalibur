#pragma once

#include "Tensor.hpp"

#include <memory>

#include <marshal_fx.hpp>
#include <tensor_builder.hpp>
#include <tensor_builder_factory.hpp>

namespace glasssix
{
    namespace excalibur
    {
        public enum class TensorLayout
        {
            RGB,
            RGBA,
            Grayscale
        };

        /// <summary>
        /// Define a tensor builder.
        /// </summary>
        public ref class TensorBuilder
        {
        public:
            TensorBuilder()
            {
                builder_ptr_ = new std::shared_ptr<tensor_builder>{ tensor_builder_factory::create() };
                builder_ = builder_ptr_->get();
            }

            ~TensorBuilder()
            {
                this->!TensorBuilder();
            }

            /// <summary>
            /// Load a bitmap from a file.
            /// </summary>
            /// <param name="path">The path of the file</param>
            void LoadFrom(System::String^ path)
            {
                CheckPointer();

                if (!builder_->load_from(marshal_fx::marshal_as<std::string>(path)))
                {
                    throw gcnew System::IO::FileNotFoundException{ "Failed to load file: " + path };
                }
            }

            /// <summary>
            /// Load a bitmap from an input stream.
            /// </summary>
            /// <param name="stream">The stream</param>
            void LoadFrom(System::IO::Stream^ stream)
            {
                CheckPointer();
                
                auto buffer = gcnew cli::array<System::Byte>(static_cast<int>(stream->Length));
                if (stream->Read(buffer, 0, buffer->Length) <= 0)
                {
                    throw gcnew System::IO::EndOfStreamException{ "The stream cannot be empty." };
                }

                LoadFrom(buffer);
            }

            /// <summary>
            /// Load a bitmap from a memory block.
            /// </summary>
            /// <param name="data">The memory block</param>
            void LoadFrom(cli::array<System::Byte>^ data)
            {
                CheckPointer();

                cli::pin_ptr<System::Byte> pinned_data = &data[0];
                auto buffer = safe_cast<System::Byte*>(pinned_data);

                if (!builder_->load_from(buffer, data->Length))
                {
                    throw gcnew System::IO::FileLoadException{ "Failed to load the stream." };
                }
            }

            /// <summary>
            /// Save the image to a file.
            /// The encoder is deduced by the file extension automatically.
            /// </summary>
            /// <param name="path">The path of the file</param>
            void SaveTo(System::String^ path)
            {
                CheckPointer();

                if (!builder_->save_to(marshal_fx::marshal_as<std::string>(path)))
                {
                    throw gcnew System::IO::IOException{ "Failed to save the image to: " + path };
                }
            }

            /// <summary>
            /// Set the parameters for building a tensor.
            /// </summary>
            /// <param name="order">The memory order</param>
            void TensorParameters(TensorOrderType order)
            {
                CheckPointer();

                builder_->tensor_parameters(static_cast<memory::orderType>(order));
            }

            /// <summary>
            /// Set the parameters for building a tensor.
            /// </summary>
            /// <param name="order">The memory order</param>
            /// <param name="device">The device ID</param>
            void TensorParameters(TensorOrderType order, int device)
            {
                CheckPointer();

                builder_->tensor_parameters(static_cast<memory::orderType>(order), device);
            }

            /// <summary>
            /// Create a floating tensor.
            /// </summary>
            /// <param name="type">The layout</param>
            /// <returns>The result</returns>
            Tensor<float>^ ToTensorSingle(TensorLayout layout)
            {
                CheckPointer();

                auto result = builder_->to_tensor_float(static_cast<tensor_layout>(layout));
                
                return result ? gcnew Tensor<float>{ System::IntPtr(&*result) } : nullptr;
            }

            /// <summary>
            /// Create a uint8 tensor.
            /// </summary>
            /// <param name="type">The layout</param>
            /// <returns>The result</returns>
            Tensor<System::Byte>^ ToTensorByte(TensorLayout layout)
            {
                CheckPointer();

                auto result = builder_->to_tensor_uint8(static_cast<tensor_layout>(layout));

                return result ? gcnew Tensor<System::Byte>{ System::IntPtr(&*result) } : nullptr;
            }
        protected:
            /// <summary>
            /// The finalizer.
            /// </summary>
            !TensorBuilder()
            {
                if (builder_ptr_ != nullptr)
                {
                    delete builder_ptr_;
                    builder_ = nullptr;
                    builder_ptr_ = nullptr;
                }
            }
        private:
            void CheckPointer()
            {
                if (builder_ == nullptr)
                {
                    throw gcnew System::NullReferenceException{ "The object has been disposed or not been initialized correctly." };
                }
            }
        private:
            tensor_builder* builder_;
            std::shared_ptr<tensor_builder>* builder_ptr_;
        };
    }
}
