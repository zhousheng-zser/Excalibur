# tensor_builder::load_from Member Function

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

Loads an image.
    </td>
    </tr>
</table>

## Overloads

<table>
    <tr>
    <td>

**[load_from(const std::string&)](#load_fromconst-stdstring)**
    </td>
    <td>

Loads an image from the disk.
    </td>
    </tr>
    <tr>
    <td>

**[load_from(std::istream&)](#load_fromstdistream)**
    </td>
    <td>

Loads an image from a **```std::istream```**.
    </td>
    </tr>
    <tr>
    <td>

**[load_from(const void*, size_t)](#load_fromconst-void-size_t)**
    </td>
    <td>

Loads an image from a memory buffer.
    </td>
    </tr>
</table>

<br>

## load_from(const std::string&)

Loads an image from the disk.

```C++
virtual bool load_from(const std::string& path) = 0;
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

## load_from(std::istream&)

Loads an image from a **```std::istream```**.

```C++
virtual bool load_from(std::istream& stream) = 0;
```
### Parameters

<dl>
    <dt>

**stream**
    </dt>
    <dd>

**```std::istream&```**
    </dd>
</dl>

The input stream.

### Returns

**```bool```**

**```true```** if the operation was successful; otherwise, **```false```** .

<br>

## load_from(const void*, size_t)

Loads an image from a memory buffer.

```C++
virtual bool load_from(const void* data, size_t size) = 0;
```

### Parameters

<dl>
    <dt>

**data**
    </dt>
    <dd>

**```const void*```**
    </dd>
</dl>

The base address of the memory buffer.

<dl>
    <dt>

**size**
    </dt>
    <dd>

**```size_t```**
    </dd>
</dl>

The length of the buffer, in bytes.

### Returns

**```bool```**

**```true```** if the operation was successful; otherwise, **```false```** .

<br>

## Links
<a href="../../Tensor_IO_Manual_en_US.md"><img src="../../images/home.png" width="32" height="32"></img></a>