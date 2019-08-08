# tensor_layout Enum

## Information

<table>
    <tr>
    <td>

Header
    </td>
    <td>
    
tensor_layout.hpp
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

Defines the layout of a **```glasssix::tensor<T>```** .
    </td>
    </tr>
</table>

## Syntax

```C++
enum class tensor_layout
{
    rgb,
    rgba,
    grayscale,
    grayscale_3
};
```

## Constants

<table>
    <tr>
    <td>

rgb
    </td>
    <td>
    
A triple-channel true color image in RGB format.
    </td>
    </tr>
    <tr>
    <td>

rgba
    </td>
    <td>
    
A quadruplet-channel true color image in RGBA format with an alpha channel.
    </td>
    </tr>
    <tr>
    <td>
        
grayscale
    </td>
    <td>

A single-channel grayscale image.
    </td>
    </tr>
    <tr>
    <td>
        
grayscale_3
    </td>
    <td>

A triple-channel grayscale image of which each pixel has three channels with identical values.
    </td>
    </tr>
</table>

## Links
<a href="../Tensor_IO_Manual_en_US.md"><img src="../images/home.png" width="32" height="32"></img></a>