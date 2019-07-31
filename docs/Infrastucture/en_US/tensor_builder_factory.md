# tensor_builder_factory Class

## Information

<table>
    <tr>
    <td>

Header
    </td>
    <td>
    
tensor_builder_factory.hpp
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

A class factory to create an instance of [**```tensor_builder```**](tensor_builder.md)
    </td>
    </tr>
</table>

## Syntax

```C++
class tensor_builder_factory final
{
public:
    static std::shared_ptr<tensor_builder> create(tensor_builder_implementation type = tensor_builder_implementation::free_image);
};
```

## Member Functions

<table>
    <tr>
    <td>

[**create(tensor_builder_implementation)**]()
    </td>
    <td>
    
Creates an instance of [**```tensor_builder```**](tensor_builder.md) .
    </td>
    </tr>
</table>

## Links
<a href="../Tensor_IO_Manual_en_US.md"><img src="../images/home.png" width="32" height="32"></img></a>