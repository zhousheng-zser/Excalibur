# tensor_builder::tensor_parameters Member Function

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

Sets the tensor parameters.
    </td>
    </tr>
</table>

## Overloads

<table>
    <tr>
    <td>

**[tensor_parameters(orderType)](#tensor_parametersorderType)**
    </td>
    <td>

Sets the tensor parameters.
    </td>
    </tr>
    <tr>
    <td>

**[tensor_parameters(orderType, int)](#tensor_parametersorderType-int)**
    </td>
    <td>

Sets the tensor parameters.
    </td>
    </tr>
</table>

<br>

## tensor_parameters(orderType)

Sets the tensor parameters.

```C++
virtual void tensor_parameters(orderType order) = 0;
```

### Parameters

<dl>
    <dt>

**order**
    </dt>
    <dd>

**```glasssix::orderType```**
    </dd>
</dl>

The order type, in standard **```glasssix::tensor```** format.

<br>

## tensor_parameters(orderType, int)

Sets the tensor parameters.

```C++
virtual void tensor_parameters(orderType order, int device) = 0;
```
### Parameters

<dl>
    <dt>

**order**
    </dt>
    <dd>

**```glasssix::orderType```**
    </dd>
</dl>

The order type, in standard **```glasssix::tensor```** format.

<dl>
    <dt>

**device**
    </dt>
    <dd>

**```int```**
    </dd>
</dl>

Indicates an physical device holding the data of the tensor.

|Value|Description|
|---|---|
|-1|Heads at CPU.|
|>= 0|Indicates the index of a GPU.|

<br>

## Links
<a href="../../Tensor_IO_Manual_en_US.md"><img src="../../images/home.png" width="32" height="32"></img></a>