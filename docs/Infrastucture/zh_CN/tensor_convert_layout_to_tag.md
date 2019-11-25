# tensor_convert_layout_to_tag 类模板

## 信息

<table>
    <tr>
    <td>

头文件
    </td>
    <td>

tensor_conversions.hpp
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

辅助类，用于转换张量的像素及通道布局。
    </td>
    </tr>
</table>

## 语法

```C++
template<tensor_layout layout>
struct tensor_convert_layout_to_tag {};
```

## 模板参数

<dl>
    <dt>

**layout**
    </dt>
    <dd>

[**```glasssix::excalibur::tensor_layout```**](tensor_layout.md)
    </dd>
</dl>

目标像素及通道布局。

## 链接
<a href="../Tensor_IO_Manual_zh_CN.md"><img src="../images/home.png" width="32" height="32"></img></a>