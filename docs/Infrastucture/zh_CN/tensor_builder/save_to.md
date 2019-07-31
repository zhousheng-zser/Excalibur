# tensor_builder::save_to 成员函数

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

将图像保存到磁盘。
    </td>
    </tr>
</table>

<br>

## save_to(const std::string&)

将图像保存到磁盘文件。

```C++
virtual bool save_to(const std::string& path) = 0;
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

磁盘文件路径。

### 返回

**```bool```**

如果保存成功，则为 **```true```**；否则为 **```false```**。

<br>

## 链接
<a href="../../Tensor_IO_Manual_zh_CN.md"><img src="../../images/home.png" width="32" height="32"></img></a>