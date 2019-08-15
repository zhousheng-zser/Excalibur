# tensor_convert_layout_to_tag Class Template

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

A helper class to be tagged with the destination layout of a tensor conversion.
    </td>
    </tr>
</table>

## Syntax

```C++
template<tensor_layout layout>
struct tensor_convert_layout_to_tag {};
```

## Template Parameters

<dl>
    <dt>

**layout**
    </dt>
    <dd>

[**```glasssix::excalibur::tensor_layout```**](tensor_layout.md)
    </dd>
</dl>

The destination layout.

## Links
<a href="../Tensor_IO_Manual_en_US.md"><img src="../images/home.png" width="32" height="32"></img></a>