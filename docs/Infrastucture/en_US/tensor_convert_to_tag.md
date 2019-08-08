# tensor_convert_to_tag Class Template

## Information

<table>
    <tr>
    <td>

Header
    </td>
    <td>

tensor_conversions.hpp
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

A helper class to be tagged with the destination underlying type of a tensor conversion.
    </td>
    </tr>
</table>

## Syntax

```C++
template<typename TDestination>
struct tensor_convert_to_tag {};
```

## Template Parameters

**```TDestination```**

The destination underlying type which must be arithmetical.

## Links
<a href="../Tensor_IO_Manual_zh_CN.md"><img src="../images/home.png" width="32" height="32"></img></a>