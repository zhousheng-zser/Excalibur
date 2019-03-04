#pragma once
#ifndef _FTENSOR_HPP_
#define _FTENSOR_HPP_

#include <glasssix\tensor.hpp>

namespace glasssix
{
	namespace excalibur
	{
		class ftensor
		{
		public:
			ftensor();
			~ftensor();
			ftensor(const std::vector<int>& shape, int device);
			ftensor(const int shape, int device = -1);
			ftensor(const ftensor& t);
			ftensor& operator=(const ftensor& t);
			ftensor clone() const;
			bool empty() const;

			tensor<float>* getdata()
			{
				return this->data;
			}

		private:
			tensor<float>* data;
		};

		inline ftensor::ftensor()
		{
			data = new tensor<float>();
		}

		inline ftensor::~ftensor()
		{
			delete data;
		}

		inline ftensor::ftensor(const std::vector<int>& shape, int device)
		{
			data = new tensor<float>(shape, device);
		}

		inline ftensor::ftensor(const int shape, int device)
		{
			data = new tensor<float>(shape, device);
		}

		inline ftensor::ftensor(const ftensor& t)
		{
			this->data = t.data;
		}

		inline ftensor& ftensor::operator=(const ftensor& t)
		{
			if (this == &t)
			{
				return *this;
			}
			this->data = t.data;
			return *this;
		}

		inline bool ftensor::empty() const
		{
			return data->empty();
		}

		inline ftensor ftensor::clone() const
		{
			ftensor ft = ftensor(data->data_shape(), data->device());
			*ft.data = this->data->clone();
			return ft;
		}

	}
}

#endif // !_FTENSOR_HPP_
