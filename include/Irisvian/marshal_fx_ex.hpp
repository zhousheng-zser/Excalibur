#pragma once

#include <marshal_fx.hpp>

namespace marshal_fx
{
    namespace details
    {
        /// <summary>
        /// An extension.
        /// </summary>
        template <typename TElement, typename TFrom>
        struct marshal_traits<cli::array<TElement>^, TFrom>
        {
            using to_type = cli::array<TElement>;

            static to_type^ marshal(const TFrom& from, const stl_collection_tag&, const tag_base&)
            {
                auto result = gcnew to_type(static_cast<int>(std::size(from)));

                int index = 0;
                for (auto& item : from)
                {
                    result[index++] = marshal_as<TElement>(item);
                }

                return result;
            }
        };

        /// <summary>
        /// An extension.
        /// </summary>
        template <typename TElement, typename TTo>
        struct marshal_traits<TTo, cli::array<TElement>^>
        {
            static TTo marshal(cli::array<TElement>^ from, const tag_base&, const stl_collection_tag&)
            {
                auto from_list = safe_cast<System::Collections::Generic::IList<TElement>^>(from);

                return marshal_as<TTo>(from_list);
            }
        };
    }
}
