# tensor_builder Class

## Information

<table>
    <tr>
    <td>

Header
    </td>
    <td>
    
tensor_builder.hpp
    </td>
    </tr>
    <tr>
    <td>

Namespace
    </td>
    <td>
    
**```glasssix::excalibur```**
    </td>
    </tr>
    <tr>
    <td>
        
Description
    </td>
    <td>

An interface class for I/O operations between standard bitmap files and standard tensors.
    </td>
    </tr>
</table>

## Syntax

```C++
class tensor_builder
{
public:
    virtual ~tensor_builder() = default;
    virtual bool load_from(const std::string& path) = 0;
    virtual bool load_from(std::istream& stream) = 0;
    virtual bool load_from(const void* data, size_t size) = 0;
    virtual bool save_to(const std::string& path) = 0;
    virtual void tensor_parameters(orderType order) = 0;
    virtual void tensor_parameters(orderType order, int device) = 0;
    virtual bool from_tensor(const tensor<float>& data, tensor_layout layout) = 0;
    virtual bool from_tensor(const tensor<uint8_t>& data, tensor_layout layout) = 0;
    virtual std::optional<tensor<float>> to_tensor_float(tensor_layout layout) = 0;
    virtual std::optional<tensor<uint8_t>> to_tensor_uint8(tensor_layout layout) = 0;
    virtual std::shared_ptr<tensor<float>> to_tensor_float_shared(tensor_layout layout) = 0;
    virtual std::shared_ptr<tensor<uint8_t>> to_tensor_uint8_shared(tensor_layout layout) = 0;
};
```

## Member Functions

<table>
    <tr>
    <td>

**[~tensor_builder()]()**
    </td>
    <td>

The default virtual destructor.
    </td>
    </tr>
    <tr>
    <td>

**[load_from(const std::string&)]()**
    </td>
    <td>

Loads an image from the disk.
    </td>
    </tr>
    <tr>
    <td>

**[load_from(const std::istream&)]()**
    </td>
    <td>

Loads an image from a **```std::istream```**.
    </td>
    </tr>
    <tr>
    <td>

**[load_from(const void*, size_t)]()**
    </td>
    <td>

Loads an image from a memory buffer.
    </td>
    </tr>
    <tr>
    <td>

**[save_to(const std::string&)]()**
    </td>
    <td>

Saves the image to a disk file.
    </td>
    </tr>
    <tr>
    <td>

**[tensor_parameters(orderType)]()**
    </td>
    <td>

Sets the tensor parameters.
    </td>
    </tr>
    <tr>
    <td>

**[tensor_parameters(orderType, int)]()**
    </td>
    <td>

Sets the tensor parameters.
    </td>
    </tr>
    <tr>
    <td>

**[from_tensor(const tensor<float>&, tensor_layout)]()**
    </td>
    <td>

Converts a floating-point tensor to a standard bitmap.
    </td>
    </tr>
    <tr>
    <td>

**[from_tensor(const tensor<uint8_t>&, tensor_layout)]()**
    </td>
    <td>

Converts a uint8 tensor to a standard bitmap.
    </td>
    </tr>
    <tr>
    <td>

**[to_tensor_float(tensor_layout)]()**
    </td>
    <td>

Converts the cached bitmap to a floating-point tensor.
    </td>
    </tr>
    <tr>
    <td>

**[to_tensor_float_shared(tensor_layout)]()**
    </td>
    <td>

Allocates a **```std::shared_ptr<tensor<float>>```**  which it converts the cached bitmap to.
    </td>
    </tr>
    <tr>
    <td>

**[to_tensor_uint8(tensor_layout)]()**
    </td>
    <td>

Converts the cached bitmap to a uint8 tensor.
    </td>
    </tr>
    <tr>
    <td>

**[to_tensor_uint8_shared(tensor_layout)]()**
    </td>
    <td>

Converts the cached bitmap to a floating-point tensor.
    </td>
    </tr>
</table>

## [![Home](../images/home.png =64x64)](../Tensor_IO_Manual_en_US.md)