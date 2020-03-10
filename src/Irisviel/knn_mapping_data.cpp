#include "knn_mapping_data.hpp"

#include <cstring>
#include <cstdint>

namespace glasssix
{
	namespace irisviel
	{
		template<int Dimension>
		struct basic_knn_mapping_data
		{
			float feature[Dimension];
			char key[33];
			bool is_active;
		};

		template<int Dimension>
		class knn_mapping_data_ref_impl : public knn_mapping_data
		{
		public:
			using data_type = basic_knn_mapping_data<Dimension>;

			knn_mapping_data_ref_impl(data_type* data) : data_{ data }
			{
			}

			virtual std::size_t size() const noexcept override
			{
				return sizeof(*data_);
			}

			virtual std::uint8_t* data() noexcept override
			{
				return reinterpret_cast<uint8_t*>(data_);
			}

			virtual const std::uint8_t* data() const noexcept override
			{
				return reinterpret_cast<const uint8_t*>(data_);
			}

			virtual bool is_active() const noexcept
			{
				return data_->is_active;
			}

			virtual void is_active(bool value) noexcept
			{
				data_->is_active = value;
			}

			virtual char* key() noexcept
			{
				return data_->key;
			}

			virtual const char* key() const noexcept
			{
				return data_->key;
			}

			virtual void key(const char* value) noexcept
			{
				std::strcpy(data_->key, value);
			}

			virtual float* feature() noexcept
			{
				return data_->feature;
			}

			virtual const float* feature() const noexcept
			{
				return data_->feature;
			}

			virtual void feature(const float* value) noexcept
			{
				std::memcpy(data_->feature, value, sizeof(data_type::feature));
			}

			virtual std::size_t feature_offset() const noexcept override
			{
				return offsetof(basic_knn_mapping_data<Dimension>, feature);
			}
		private:
			data_type* data_;
		};

		template<int Dimension>
		class knn_mapping_data_impl : public knn_mapping_data
		{
		public:
			using data_type = basic_knn_mapping_data<Dimension>;

			knn_mapping_data_impl() : data_{ }
			{
			}

			knn_mapping_data_impl(const data_type& data) : data_{ data }
			{
			}

			knn_mapping_data_impl(data_type&& data) : data_{ data }
			{
			}

			virtual std::size_t size() const noexcept override
			{
				return sizeof(data_);
			}

			virtual std::uint8_t* data() noexcept override
			{
				return reinterpret_cast<uint8_t*>(&data_);
			}

			virtual const std::uint8_t* data() const noexcept override
			{
				return reinterpret_cast<const uint8_t*>(&data_);
			}

			virtual bool is_active() const noexcept
			{
				return data_.is_active;
			}

			virtual void is_active(bool value) noexcept
			{
				data_.is_active = value;
			}

			virtual char* key() noexcept
			{
				return data_.key;
			}

			virtual const char* key() const noexcept
			{
				return data_.key;
			}

			virtual void key(const char* value) noexcept
			{
				std::strcpy(data_.key, value);
			}

			virtual float* feature() noexcept
			{
				return data_.feature;
			}

			virtual const float* feature() const noexcept
			{
				return data_.feature;
			}

			virtual void feature(const float* value) noexcept
			{
				std::memcpy(data_.feature, value, sizeof(data_type::feature));
			}

			virtual std::size_t feature_offset() const noexcept override
			{
				return offsetof(data_type, feature);
			}
		private:
			data_type data_;
		};

		std::shared_ptr<knn_mapping_data> knn_mapping_data::shared()
		{
			return shared_from_this();
		}

		std::size_t knn_mapping_data::struct_size(int dimension) noexcept
		{
			switch (dimension)
			{
			case 128:
				return sizeof(knn_mapping_data_impl<128>);
			case 512:
				return sizeof(knn_mapping_data_impl<512>);
			default:
				return sizeof(knn_mapping_data_impl<128>);
			}
		}

		std::size_t knn_mapping_data::feature_offset(int dimension) noexcept
		{
			switch (dimension)
			{
			case 128:
				return offsetof(basic_knn_mapping_data<128>, feature);
			case 512:
				return offsetof(basic_knn_mapping_data<512>, feature);
			default:
				return offsetof(basic_knn_mapping_data<128>, feature);
			}
		}

		std::shared_ptr<knn_mapping_data> knn_mapping_data::create(int dimension)
		{
			switch (dimension)
			{
			case 128:
				return std::make_shared<knn_mapping_data_impl<128>>();
			case 512:
				return std::make_shared<knn_mapping_data_impl<512>>();
			default:
				return std::make_shared<knn_mapping_data_impl<128>>();
			}
		}

		std::shared_ptr<knn_mapping_data> knn_mapping_data::create(int dimension, std::uint8_t* ptr)
		{
			switch (dimension)
			{
			case 128:
				return std::make_shared<knn_mapping_data_impl<128>>(*reinterpret_cast<basic_knn_mapping_data<128>*>(ptr));
			case 512:
				return std::make_shared<knn_mapping_data_impl<512>>(*reinterpret_cast<basic_knn_mapping_data<512>*>(ptr));
			default:
				return std::make_shared<knn_mapping_data_impl<128>>(*reinterpret_cast<basic_knn_mapping_data<128>*>(ptr));
			}
		}

		std::shared_ptr<knn_mapping_data> knn_mapping_data::create_ref(int dimension, std::uint8_t* ptr)
		{
			switch (dimension)
			{
			case 128:
				return std::make_shared<knn_mapping_data_ref_impl<128>>(reinterpret_cast<basic_knn_mapping_data<128>*>(ptr));
			case 512:
				return std::make_shared<knn_mapping_data_ref_impl<512>>(reinterpret_cast<basic_knn_mapping_data<512>*>(ptr));
			default:
				return std::make_shared<knn_mapping_data_ref_impl<128>>(reinterpret_cast<basic_knn_mapping_data<128>*>(ptr));
			}
		}

		bool knn_mapping_data::key_equals(const knn_mapping_data& left, const knn_mapping_data& right)
		{
			return !strcmp(left.key(), right.key());
		}
	}
}
