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

**[load_from(const std::string&)](#load_from%28const%20std::string&%29)**
    </td>
    <td>

从磁盘加载一个图像文件。
    </td>
    </tr>
    <tr>
    <td>

**[load_from(const std::istream&)]()**
    </td>
    <td>

从 **```std::istream```** 加载一个图像。
    </td>
    </tr>
    <tr>
    <td>

**[load_from(const void*, size_t)]()**
    </td>
    <td>

从内存缓冲区加载一个图像。
    </td>
    </tr>
</table>

## load_from(const std::string&)

从磁盘加载一个图像文件。

```C++
virtual bool load_from(const std::string& path) = 0;
```

### 参数

<dl display="flex">
    <dt>

**path**
    </dt>
    <dd>

**```std::string```**
    </dd>
</dl>

图像文件的路径。

## 链接
<a href="../Tensor_IO_Manual_zh_CN.md"><img src="../images/home.png" width="32" height="32"></img></a>
