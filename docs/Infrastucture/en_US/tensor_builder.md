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

|   |   |
|---|---|
|Header|tensor_builder.hpp|
|Namespace|**```glasssix::excalibur```**|
|Description|An interface class for I/O operations between standard bitmap files and standard tensors.|

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

|||
|---|---|
|**[~tensor_builder()]()**|The default virtual destructor.|
|**[load_from(const std::string&)]()**|Loads an image from the disk.|
|**[load_from(const std::istream&)]()**|Loads an image from a **```std::istream```**.|
|**[load_from(const void*, size_t)]()**|Loads an image from a memory buffer.|
|**[save_to(const std::string&)]()**|Saves the image to a disk file.|
|**[tensor_parameters(orderType)]()**|Sets the tensor parameters.|
|**[tensor_parameters(orderType, int)]()**|Sets the tensor parameters.|
|**[from_tensor(const tensor<float>&, tensor_layout)]()**|Converts a floating-point tensor to a standard bitmap.|
|**[from_tensor(const tensor<uint8_t>&, tensor_layout)]()**|Converts a uint8 tensor to a standard bitmap.|
|**[to_tensor_float(tensor_layout)]()**|Converts the cached bitmap to a floating-point tensor.|
|**[to_tensor_float_shared(tensor_layout)]()**|Allocates a **```std::shared_ptr<tensor<float>>```**  which it converts the cached bitmap to.|
|**[to_tensor_uint8(tensor_layout)]()**|Converts the cached bitmap to a uint8 tensor.|
|**[to_tensor_uint8_shared(tensor_layout)]()**|Allocates a **```std::shared_ptr<tensor<uint8_t>>```**  which it converts the cached bitmap to.|

## [Return](../Tensor_IO_Manual_en_US.md)
