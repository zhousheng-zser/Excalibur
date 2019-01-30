#pragma once
#ifndef _UTENSOR_HPP_
#define _UTENSOR_HPP_

#include <glasssix\tensor.hpp>

namespace glasssix
{
	namespace excalibur
	{
		class utensor
		{
		public:
			utensor();
			~utensor();
			utensor(const std::vector<int>& shape, int device);
			utensor(const int shape, int device = -1);
			utensor(const utensor& t);
			utensor& operator=(const utensor& t);
			utensor clone() const;
			bool empty() const;

			tensor<unsigned char>* getdata()
			{
				return this->data;
			}

		private:
			tensor<unsigned char>* data;
		};

		inline utensor::utensor()
		{
			data = new tensor<unsigned char>();
		}

		inline utensor::~utensor()
		{
			delete data;
		}

		inline utensor::utensor(const std::vector<int>& shape, int device)
		{
			data = new tensor<unsigned char>(shape, device);
		}

		inline utensor::utensor(const int shape, int device)
		{
			data = new tensor<unsigned char>(shape, device);
		}

		inline utensor::utensor(const utensor& t)
		{
			this->data = t.data;
		}

		inline utensor& utensor::operator=(const utensor& t)
		{
			if (this == &t)
			{
				return *this;
			}
			this->data = t.data;
			return *this;
		}

		inline bool utensor::empty() const
		{
			return data->empty();
		}

		inline utensor utensor::clone() const
		{
			utensor ft = utensor(data->data_shape(), data->device());
			*ft.data = this->data->clone();
			return ft;
		}

	}
}

#endif // !_UTENSOR_HPP_
