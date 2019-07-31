# tensor_builder::load_from 成员函数

## 信息

<table>
    <tr>
    <td>

头文件
    </td>
    <td>

tensor_builder.hpp
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

加载一个图像。
    </td>
    </tr>
</table>

## 重载

<table>
    <tr>
    <td>

**[load_from(const std::string&)](#load_fromconst-stdstring)**
    </td>
    <td>

从磁盘加载一个图像文件。
    </td>
    </tr>
    <tr>
    <td>

**[load_from(std::istream&)](#load_fromstdistream)**
    </td>
    <td>

从 **```std::istream```** 加载一个图像。
    </td>
    </tr>
    <tr>
    <td>

**[load_from(const void*, size_t)](#load_fromconst-void-size_t)**
    </td>
    <td>

从内存缓冲区加载一个图像。
    </td>
    </tr>
</table>

<br>

## load_from(const std::string&)

从磁盘加载一个图像文件。

```C++
virtual bool load_from(const std::string& path) = 0;
```

### 参数

<dl>
    <dt>

**path**
    </dt>
    <dd>

**```const std::string&```**
    </dd>
</dl>

图像文件的路径。

### 返回

**```bool```**

如果加载成功，则为 **```true```**；否则为 **```false```**。

<br>

## load_from(std::istream&)

从 **```std::istream```** 加载一个图像。

```C++
virtual bool load_from(std::istream& stream) = 0;
```
### 参数

<dl>
    <dt>

**stream**
    </dt>
    <dd>

**```std::istream&```**
    </dd>
</dl>

输入流。

### 返回

**```bool```**

如果加载成功，则为 **```true```**；否则为 **```false```**。

<br>

## load_from(const void*, size_t)

从磁盘加载一个图像文件。

```C++
virtual bool load_from(const void* data, size_t size) = 0;
```

### 参数

<dl>
    <dt>

**data**
    </dt>
    <dd>

**```const void*```**
    </dd>
</dl>

内存缓冲区的首地址。

<dl>
    <dt>

**size**
    </dt>
    <dd>

**```size_t```**
    </dd>
</dl>

内存缓冲区的字节长度。

### 返回

**```bool```**

如果加载成功，则为 **```true```**；否则为 **```false```**。

<br>

## 链接
<a href="../../Tensor_IO_Manual_zh_CN.md"><img src="../../images/home.png" width="32" height="32"></img></a>
