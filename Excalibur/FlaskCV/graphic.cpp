#include "graphic.hpp"

namespace flaskcv
{
	graphic::graphic(): c_(0), h_(0), w_(0), device_(0)
	{
		data_ = nullptr;
	}


	graphic::~graphic()
	{
		delete data_;
	}

	graphic::graphic(int c): c_(c), h_(1), w_(1), device_(-1)
	{
		data_ = new syncedmem(c_ * sizeof(float), device_);
	}

	graphic::graphic(int h, int w): c_(1), h_(h), w_(w), device_(-1)
	{
		data_ = new syncedmem(h_*w_ * sizeof(float), device_);
	}

	graphic::graphic(int c, int h, int w): c_(c), h_(h), w_(w), device_(-1)
	{
		data_ = new syncedmem(c_* h_*w_ * sizeof(float), device_);
	}

	graphic::graphic(int c, float* data): c_(c), h_(1), w_(1), device_(-1)
	{
		data_ = new syncedmem(c_ * sizeof(float), device_);
		data_->set_cpu_data(data);
	}

	graphic::graphic(int h, int w, float* data): c_(1), h_(h), w_(w), device_(-1)
	{
		data_ = new syncedmem(h_*w_ * sizeof(float), device_);
		data_->set_cpu_data(data);
	}

	graphic::graphic(int c, int h, int w, float* data): c_(c), h_(h), w_(w), device_(-1)
	{
		data_ = new syncedmem(c_* h_*w_ * sizeof(float), device_);
		data_->set_cpu_data(data);
	}

	graphic::graphic(const graphic& g): data_(g.data_), c_(g.c_), h_(g.h_), w_(g.w_), device_(-1)
	{
		
	}

	graphic& graphic::operator=(const graphic& g)
	{
		if (this==&g)
		{
			return *this;
		}
		if (data_!=nullptr)
		{
			delete data_;
		}
		data_ = g.data_;

		c_ = g.c_;
		h_ = g.h_;
		w_ = g.w_;

		return *this;
	}

	graphic graphic::clone() const
	{
		if (empty())
		{
			return graphic();
		}
		graphic g(c_, h_, w_);
		memcpy(g.data_, data_, g.count() * sizeof(float));
		return g;
	}

	int graphic::count() const
	{
		return c_*h_*w_;
	}

	bool graphic::empty() const
	{
		return data_ == nullptr || count() == 0;
	}

	void graphic::fill(float x)
	{
		float* cpu_data = static_cast<float*>(data_->mutable_cpu_data());
		for (int i = 0; i < count(); i++)
		{
			cpu_data[i] = x;
		}
	}

	int graphic::channel() const
	{
		return c_;
	}


	int graphic::height() const
	{
		return h_;
	}

	int graphic::width() const
	{
		return w_;
	}

	graphic graphic::channel_graphic_ptr(int c)
	{
		return graphic(h_, w_, this->mutable_cpu_data() + h_ * w_ * c);
	}


	const float* graphic::cpu_data() const
	{
		return static_cast<const float*>(data_->cpu_data());
	}

	float* graphic::mutable_cpu_data() const
	{
		return static_cast<float*>(data_->mutable_cpu_data());
	}

	
}
