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

**[load_from(const std::string&)](tensor_builder/load_from.md#load_fromconst-stdstring)**
    </td>
    <td>

Loads an image from the disk.
    </td>
    </tr>
    <tr>
    <td>

**[load_from(std::istream&)](tensor_builder/load_from.md#load_fromstdistream)**
    </td>
    <td>

Loads an image from a **```std::istream```**.
    </td>
    </tr>
    <tr>
    <td>

**[load_from(const void*, size_t)](tensor_builder/load_from.md#load_fromconst-void-size_t)**
    </td>
    <td>

Loads an image from a memory buffer.
    </td>
    </tr>
    <tr>
    <td>

**[save_to(const std::string&)](tensor_builder/save_to.md#save_toconst-stdstring)**
    </td>
    <td>

Saves the image to a disk file.
    </td>
    </tr>
    <tr>
    <td>

**[tensor_parameters(orderType)](tensor_builder/tensor_parameters.md#tensor_parametersorderType)**
    </td>
    <td>

Sets the tensor parameters.
    </td>
    </tr>
    <tr>
    <td>

**[tensor_parameters(orderType, int)](tensor_builder/tensor_parameters.md#tensor_parametersorderType-int)**
    </td>
    <td>

Sets the tensor parameters.
    </td>
    </tr>
    <tr>
    <td>

**[from_tensor(const tensor<float>&, tensor_layout)](tensor_builder/from_tensor.md#from_tensorconst-tensorfloat-tensor_layout)**
    </td>
    <td>

Converts a floating-point tensor to a standard bitmap.
    </td>
    </tr>
    <tr>
    <td>

**[from_tensor(const tensor<uint8_t>&, tensor_layout)](tensor_builder/from_tensor.md#from_tensorconst-tensoruint8_t-tensor_layout)**
    </td>
    <td>

Converts a uint8 tensor to a standard bitmap.
    </td>
    </tr>
    <tr>
    <td>

**[to_tensor_float(tensor_layout)](tensor_builder/to_tensor_float.md#to_tensor_floattensor_layout)**
    </td>
    <td>

Converts the loaded bitmap to a floating-point tensor.
    </td>
    </tr>
    <tr>
    <td>

**[to_tensor_float_shared(tensor_layout)](tensor_builder/to_tensor_float_shared.md#to_tensor_float_sharedtensor_layout)**
    </td>
    <td>

Allocates a **```std::shared_ptr<tensor<float>>```**  which it converts the loaded bitmap to.
    </td>
    </tr>
    <tr>
    <td>

**[to_tensor_uint8(tensor_layout)](tensor_builder/to_tensor_uint8.md#to_tensor_uint8tensor_layout)**
    </td>
    <td>

Converts the loaded bitmap to a uint8 tensor.
    </td>
    </tr>
    <tr>
    <td>

**[to_tensor_uint8_shared(tensor_layout)](tensor_builder/to_tensor_uint8_shared.md#to_tensor_uint8_sharedtensor_layout)**
    </td>
    <td>

Allocates a **```std::shared_ptr<tensor<uint8_t>>```**  which it converts the loaded bitmap to.
    </td>
    </tr>
</table>

## Links
<a href="../Tensor_IO_Manual_en_US.md"><img src="../images/home.png" width="32" height="32"></img></a>