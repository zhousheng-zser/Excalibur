# tensor_builder_factory 类

## 信息

<table>
    <tr>
    <td>

头文件
    </td>
    <td>

tensor_builder_factory.hpp
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

类工厂，用于创建 [**```tensor_builder```**](tensor_builder.md) 的实例。
    </td>
    </tr>
</table>

## 语法

```C++
class tensor_builder_factory final
{
public:
    static std::shared_ptr<tensor_builder> create(tensor_builder_implementation type = tensor_builder_implementation::free_image);
};
```

## 成员函数

<table>
    <tr>
    <td>

[**create(tensor_builder_implementation)**]()
    </td>
    <td>

创建一个 [**```tensor_builder```**](tensor_builder.md) 的实例。
    </td>
    </tr>
</table>

## 链接
<a href="../Tensor_IO_Manual_zh_CN.md"><img src="../images/home.png" width="32" height="32"></img></a>
