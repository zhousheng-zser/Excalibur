# tensor_builder::to_tensor_uint8 Member Function

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

Converts the loaded bitmap to a uint8 tensor.
    </td>
    </tr>
</table>

<br>

## to_tensor_uint8(tensor_layout)

Converts the loaded bitmap to a uint8 tensor.

```C++
virtual std::optional<tensor<uint8_t>> to_tensor_uint8(tensor_layout layout) = 0;
```

### Parameters

<dl>
    <dt>

**layout**
    </dt>
    <dd>

[**```glasssix::excalibur::tensor_layout```**](../tensor_layout.md)
    </dd>
</dl>

Indicates the channel count and the pixel format.

### Returns

**```std::optional<glasssix::tensor<uint8_t>>```**

An optional object which contains the converted tensor if the operation was successful and otherwise, an empty value is included.

<br>

## Links
<a href="../../Tensor_IO_Manual_en_US.md"><img src="../../images/home.png" width="32" height="32"></img></a>