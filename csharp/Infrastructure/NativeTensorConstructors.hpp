#pragma once

#include "TensorOrderType.hpp"

#include <marshal_fx.hpp>

namespace glasssix
{
    namespace excalibur
    {
        struct tensor_;

        /// <summary>
        /// A helper class to make abstraction of native tensors.
        /// </summary>
        ref class NativeTensorConstructors
        {
        public:
            /// <summary>
            /// Create an instance of NativeTensorConstructors.
            /// </summary>
            /// <param name="code">The type code</param>
            NativeTensorConstructors(System::TypeCode code)
            {
                switch (code)
                {
                case System::TypeCode::Byte:
                    InitializeConstructorsCore<uint8_t>();
                    break;
                case System::TypeCode::SByte:
                    InitializeConstructorsCore<int8_t>();
                    break;
                case System::TypeCode::Int16:
                    InitializeConstructorsCore<int16_t>();
                    break;
                case System::TypeCode::Int32:
                    InitializeConstructorsCore<int32_t>();
                    break;
                case System::TypeCode::Int64:
                    InitializeConstructorsCore<int64_t>();
                    break;
                case System::TypeCode::UInt16:
                    InitializeConstructorsCore<uint16_t>();
                    break;
                case System::TypeCode::UInt32:
                    InitializeConstructorsCore<uint32_t>();
                    break;
                case System::TypeCode::UInt64:
                    InitializeConstructorsCore<uint64_t>();
                    break;
                case System::TypeCode::Single:
                    InitializeConstructorsCore<float>();
                    break;
                case System::TypeCode::Double:
                    InitializeConstructorsCore<double>();
                    break;
                default:
                    throw gcnew System::ArgumentException("Unsupported type.");
                }
            }

            /// <summary>
            /// Lookup a constructor.
            /// </summary>
            /// <param name="method">The method information</param>
            /// <returns>The delegate of the constructor</returns>
            System::Delegate^ Lookup(System::Reflection::MethodBase^ method)
            {
                System::Delegate^ result;
                auto sig = GetFunctionSig(method->GetParameters());

                return constructors_->TryGetValue(sig, result) ? result : nullptr;
            }

            /// <summary>
            /// Retrieve the function signiture of a method.
            /// </summary>
            /// <param name="params">The type array</param>
            /// <returns>The function sign</returns>
            static System::String^ GetFunctionSig(...cli::array<System::Type^>^ params)
            {
                return System::String::Concat(params);
            }

            /// <summary>
            /// Retrieve the function signiture of a method.
            /// </summary>
            /// <param name="params">The parameter array</param>
            /// <returns>The function sign</returns>
            static System::String^ GetFunctionSig(cli::array<System::Reflection::ParameterInfo^>^ params)
            {
                auto list = safe_cast<System::Collections::Generic::IList<System::Reflection::ParameterInfo^>^>(params);
                auto values = System::Linq::Enumerable::Select(list, gcnew System::Func<System::Reflection::ParameterInfo^, System::String^>(ToStringCore));

                return System::String::Concat(values);
            }
        private:
            /// <summary>
            /// Initialize the constructors.
            /// </summary>
            template<typename TUnderlyingType>
            void InitializeConstructorsCore()
            {
                auto suffix = System::String::Format("<{0}>", gcnew System::String{ typeid(TUnderlyingType).name() });

                constructors_ = gcnew System::Collections::Generic::Dictionary<System::String^, System::Delegate^>{};

                // Reference all constructors manually to force the compiler to instantiate the template functions.
                reference(ConstructorAlpha<TUnderlyingType>);
                reference(ConstructorBeta<TUnderlyingType>);
                reference(ConstructorGama<TUnderlyingType>);

                // Get all constructors.
                auto type = GetType();
                auto methods = type->GetMethods(System::Reflection::BindingFlags::NonPublic | System::Reflection::BindingFlags::Static);
                for each (auto item in methods)
                {
                    if (item->Name->StartsWith("Constructor") && item->Name->EndsWith(suffix, System::StringComparison::OrdinalIgnoreCase))
                    {
                        auto sig = GetFunctionSig(item->GetParameters());
                        auto handler = GetDelegateFromStaticCore(item);
                        constructors_[sig] = handler;
                    }
                }
            }

            /// <summary>
            /// Create a delegate from a static method.
            /// </summary>
            /// <param name="method">The static method</param>
            /// <returns>The delegate</returns>
            static System::Delegate^ GetDelegateFromStaticCore(System::Reflection::MethodInfo^ method)
            {
                using namespace System::Linq;
                using namespace System::Linq::Expressions;
                using namespace System::Collections::Generic;

                auto params = method->GetParameters();
                auto void_method = method->ReturnType == System::Void::typeid;
                auto list = safe_cast<IList<System::Reflection::ParameterInfo^>^>(params);
                auto get_delegate_type = void_method ?
                    gcnew System::Func<cli::array<System::Type^>^, System::Type^>(&Expression::GetActionType) :
                    gcnew System::Func<cli::array<System::Type^>^, System::Type^>(&Expression::GetFuncType);

                auto parameter_types = Enumerable::Select(list, gcnew System::Func<System::Reflection::ParameterInfo^, System::Type^>(ToTypeCore));

                // Add the return type if necessary.
                if (!void_method)
                {
                    parameter_types = Enumerable::Concat(parameter_types, safe_cast<IList<System::Type^>^>(gcnew cli::array<System::Type^>{ method->ReturnType }));
                }
               
                return method->CreateDelegate(get_delegate_type(Enumerable::ToArray(parameter_types)));
            }

            /// <summary>
            /// A helper method for converting System::Reflection::ParameterInfo to System::Type.
            /// </summary>
            /// <param name="info">The parameter information</param>
            /// <returns>The type</returns>
            static System::Type^ ToTypeCore(System::Reflection::ParameterInfo^ info)
            {
                return info->ParameterType;
            }

            /// <summary>
            /// A helper method for converting System::Reflection::ParameterInfo to System::String.
            /// </summary>
            /// <param name="info">The parameter information</param>
            /// <returns>The string</returns>
            static System::String^ ToStringCore(System::Reflection::ParameterInfo^ info)
            {
                return info->ParameterType->ToString();
            }
        private:
            template<typename TUnderlyingType>
            static System::IntPtr ConstructorAlpha(TensorOrderType order)
            {
                return System::IntPtr{ new tensor<TUnderlyingType>{ static_cast<orderType>(order) } };
            }

            template<typename TUnderlyingType>
            static System::IntPtr ConstructorBeta(int shape, int device, TensorOrderType order)
            {
                return System::IntPtr{ new tensor<TUnderlyingType>{ shape, device, static_cast<orderType>(order) } };
            }

            template<typename TUnderlyingType>
            static System::IntPtr ConstructorGama(System::Collections::Generic::IList<int>^ shape, int device, TensorOrderType order)
            {
                auto native_shape = marshal_fx::marshal_as<std::vector<int>>(shape);

                return System::IntPtr{ new tensor<TUnderlyingType>{ native_shape, device, static_cast<orderType>(order) } };
            }

            template<typename TFunc>
            constexpr static void reference(TFunc&& func)
            {
                auto dummy = func;
            }
        private:
            System::Collections::Generic::Dictionary<System::String^, System::Delegate^>^ constructors_;
        };
    }
}
