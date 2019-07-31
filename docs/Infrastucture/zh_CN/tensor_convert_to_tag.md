# tensor_convert_to_tag 类模板

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

辅助类，用于标记张量基础类型转换的目标类型。
    </td>
    </tr>
</table>

## 语法

```C++
template<typename TDestination>
struct tensor_convert_to_tag {};
```

## 模板参数

**```TDestination```**

目标基础类型，必须为算术类型。

## 链接
<a href="../Tensor_IO_Manual_zh_CN.md"><img src="../images/home.png" width="32" height="32"></img></a>