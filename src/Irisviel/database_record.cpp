#include "database_record.hpp"

#include <cstring>
#include <cstdint>

namespace glasssix
{
	namespace irisviel
	{
		template<int Dimension>
		struct basic_database_record
		{
			float feature[Dimension];
			char key[33];
			bool active;
		};

		template<int Dimension>
		class database_record_ref_impl : public database_record
		{
		public:
			using data_type = basic_database_record<Dimension>;

			database_record_ref_impl(data_type* data) : data_{ data }
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

			virtual bool active() const noexcept override
			{
				return data_->active;
			}

			virtual void active(bool value) noexcept override
			{
				data_->active = value;
			}

			virtual char* key() noexcept override
			{
				return data_->key;
			}

			virtual const char* key() const noexcept override
			{
				return data_->key;
			}

			virtual void key(const char* value) noexcept override
			{
				std::strcpy(data_->key, value);
			}

			virtual float* feature() noexcept override
			{
				return data_->feature;
			}

			virtual const float* feature() const noexcept override
			{
				return data_->feature;
			}

			virtual void feature(const float* value) noexcept override
			{
				std::memcpy(data_->feature, value, sizeof(data_type::feature));
			}

			virtual std::size_t feature_offset() const noexcept override
			{
				return offsetof(basic_database_record<Dimension>, feature);
			}
		private:
			data_type* data_;
		};

		template<int Dimension>
		class database_record_impl : public database_record
		{
		public:
			using data_type = basic_database_record<Dimension>;

			database_record_impl() : data_{ }
			{
			}

			database_record_impl(const data_type& data) : data_{ data }
			{
			}

			database_record_impl(data_type&& data) : data_{ data }
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

			virtual bool active() const noexcept override
			{
				return data_.active;
			}

			virtual void active(bool value) noexcept override
			{
				data_.active = value;
			}

			virtual char* key() noexcept override
			{
				return data_.key;
			}

			virtual const char* key() const noexcept override
			{
				return data_.key;
			}

			virtual void key(const char* value) noexcept override
			{
				std::strcpy(data_.key, value);
			}

			virtual float* feature() noexcept override
			{
				return data_.feature;
			}

			virtual const float* feature() const noexcept override
			{
				return data_.feature;
			}

			virtual void feature(const float* value) noexcept override
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

		std::shared_ptr<database_record> database_record::shared()
		{
			return shared_from_this();
		}

		std::size_t database_record::struct_size(int dimension) noexcept
		{
			switch (dimension)
			{
			case 128:
				return sizeof(database_record_impl<128>::data_type);
			case 512:
				return sizeof(database_record_impl<512>::data_type);
			default:
				return sizeof(database_record_impl<128>::data_type);
			}
		}

		std::size_t database_record::feature_offset(int dimension) noexcept
		{
			switch (dimension)
			{
			case 128:
				return offsetof(basic_database_record<128>, feature);
			case 512:
				return offsetof(basic_database_record<512>, feature);
			default:
				return offsetof(basic_database_record<128>, feature);
			}
		}

		std::shared_ptr<database_record> database_record::create(int dimension)
		{
			switch (dimension)
			{
			case 128:
				return std::make_shared<database_record_impl<128>>();
			case 512:
				return std::make_shared<database_record_impl<512>>();
			default:
				return std::make_shared<database_record_impl<128>>();
			}
		}

		std::shared_ptr<database_record> database_record::create(int dimension, std::uint8_t* ptr)
		{
			switch (dimension)
			{
			case 128:
				return std::make_shared<database_record_impl<128>>(*reinterpret_cast<basic_database_record<128>*>(ptr));
			case 512:
				return std::make_shared<database_record_impl<512>>(*reinterpret_cast<basic_database_record<512>*>(ptr));
			default:
				return std::make_shared<database_record_impl<128>>(*reinterpret_cast<basic_database_record<128>*>(ptr));
			}
		}

		std::shared_ptr<database_record> database_record::create_ref(int dimension, std::uint8_t* ptr)
		{
			switch (dimension)
			{
			case 128:
				return std::make_shared<database_record_ref_impl<128>>(reinterpret_cast<basic_database_record<128>*>(ptr));
			case 512:
				return std::make_shared<database_record_ref_impl<512>>(reinterpret_cast<basic_database_record<512>*>(ptr));
			default:
				return std::make_shared<database_record_ref_impl<128>>(reinterpret_cast<basic_database_record<128>*>(ptr));
			}
		}

		bool database_record::key_equals(const char* left, const char* right)
		{
			return !_stricmp(left, right);
		}

		bool database_record::key_equals(const database_record& left, const database_record& right)
		{
			return key_equals(left.key(), right.key());
		}
	}
}
