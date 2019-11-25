# tensor_convert_to 变量模板

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

对 [**```tensor_convert_to_tag```**](tensor_convert_to_tag.md) 的包装。
    </td>
    </tr>
</table>

## 语法

```C++
template<tensor_layout layout>
tensor_convert_layout_to_tag<layout> tensor_convert_layout_to;
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