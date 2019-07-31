# tensor_builder 类

## 信息

<table>
    <tr>
    <td>

头文件
    </td>
    <td>
    
tensor_builder.hpp
    </td>
    </tr>
    <tr>
    <td>

命名空间
    </td>
    <td>
    
**```glasssix::excalibur```**
    </td>
    </tr>
    <tr>
    <td>
        
功能描述
    </td>
    <td>

提供标准位图和张量间的 I/O 操作。
    </td>
    </tr>
</table>

## 语法

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

## 成员函数

<table>
    <tr>
    <td>

**[~tensor_builder()]()**
    </td>
    <td>

默认虚析构函数。
    </td>
    </tr>
    <tr>
    <td>

**[load_from(const std::string&)](tensor_builder/load_from.md#load_fromconst-stdstring)**
    </td>
    <td>

从磁盘加载一个图像文件。
    </td>
    </tr>
    <tr>
    <td>

**[load_from(std::istream&)](tensor_builder/load_from.md#load_fromstdistream)**
    </td>
    <td>

从 **```std::istream```** 加载一个图像。
    </td>
    </tr>
    <tr>
    <td>

**[load_from(const void*, size_t)](tensor_builder/load_from.md#load_fromconst-void-size_t)**
    </td>
    <td>

从内存缓冲区加载一个图像。
    </td>
    </tr>
    <tr>
    <td>

**[save_to(const std::string&)](tensor_builder/save_to.md#save_toconst-stdstring)**
    </td>
    <td>

将图像保存到磁盘文件。
    </td>
    </tr>
    <tr>
    <td>

**[tensor_parameters(orderType)](tensor_builder/tensor_parameters.md#tensor_parametersorderType)**
    </td>
    <td>

设置标准张量的参数。
    </td>
    </tr>
    <tr>
    <td>

**[tensor_parameters(orderType, int)](tensor_builder/tensor_parameters.md#tensor_parametersorderType-int)**
    </td>
    <td>

设置标准张量的参数。
    </td>
    </tr>
    <tr>
    <td>

**[from_tensor(const tensor<float>&, tensor_layout)](tensor_builder/from_tensor.md#from_tensorconst-tensorfloat-tensor_layout)**
    </td>
    <td>

将一个浮点型张量转换为标准位图。
    </td>
    </tr>
    <tr>
    <td>

**[from_tensor(const tensor<uint8_t>&, tensor_layout)](tensor_builder/from_tensor.md#from_tensorconst-tensor<uint8_t-tensor_layout)**
    </td>
    <td>

将一个字节型张量转换为标准位图。
    </td>
    </tr>
    <tr>
    <td>

**[to_tensor_float(tensor_layout)](tensor_builder/to_tensor_float.md#to_tensor_floattensor_layout)**
    </td>
    <td>

将已加载的标准位图转换为浮点型张量。
    </td>
    </tr>
    <tr>
    <td>

**[to_tensor_float_shared(tensor_layout)](tensor_builder/to_tensor_float_shared.md#to_tensor_float_sharedtensor_layout)**
    </td>
    <td>

申请一个 **```std::shared_ptr<tensor<float>>```**  对象并将已加载的标准位图转换为浮点型张量，存储于该对象中。
    </td>
    </tr>
    <tr>
    <td>

**[to_tensor_uint8(tensor_layout)](tensor_builder/to_tensor_uint8.md#to_tensor_uint8tensor_layout)**
    </td>
    <td>

将已加载的标准位图转换为字节型张量。
    </td>
    </tr>
    <tr>
    <td>

**[to_tensor_uint8_shared(tensor_layout)](tensor_builder/to_tensor_uint8_shared.md#to_tensor_uint8_sharedtensor_layout)**
    </td>
    <td>

申请一个 **```std::shared_ptr<tensor<uint8>>```**  对象，并将已加载的标准位图转换为字节型张量，存储于该对象中。
    </td>
    </tr>
</table>

## 链接
<a href="../Tensor_IO_Manual_zh_CN.md"><img src="../images/home.png" width="32" height="32"></img></a>