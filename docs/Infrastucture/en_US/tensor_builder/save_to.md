# tensor_builder::save_to Member Function

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

Saves the image to the disk.
    </td>
    </tr>
</table>

<br>

## save_to(const std::string&)

Saves the image to a disk file.

```C++
virtual bool save_to(const std::string& path) = 0;
```

### Parameters

<dl>
    <dt>

**path**
    </dt>
    <dd>

**```const std::string&```**
    </dd>
</dl>

The path of the image file.

### Returns

**```bool```**

**```true```** if the operation was successful; otherwise, **```false```** .

<br>

## Links
<a href="../../Tensor_IO_Manual_en_US.md"><img src="../../images/home.png" width="32" height="32"></img></a>