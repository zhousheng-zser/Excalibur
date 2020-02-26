using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace G6.Algorithm.Lib
{
    public class lib_gaiunia
    {
        /// <summary>
        /// 获取版本信息
        /// </summary>
        /// <returns>版本字符串首地址, 非托管内存需释放</returns>
        [DllImport(@"libGaius")]
        public static extern IntPtr Gaius_getVersion();
        /// <summary>
        /// 获取实例
        /// </summary>
        /// <param name="device">使用的GPU索引，-1表示使用CPU</param>
        /// <returns>实例的指针</returns>
        [DllImport(@"libGaius")]
        public static extern IntPtr Gaius_NewInstance(int device);

        /// <summary>
        /// 释放实例
        /// </summary>
        /// <param name="instance">实例的指针</param>
        [DllImport(@"libGaius")]
        public static extern void Gaius_ReleaseInstance(IntPtr instance);

        /// <summary>
        /// 特征提取
        /// </summary>
        /// <param name="instance">实例的指针</param>
        /// <param name="faces">对齐后的图片的首地址</param>
        /// <param name="num">人脸数</param>
        /// <param name="order">一般为0(NCHW)</param>
        /// <returns>输出的特征向量的首地址, 向量数据类型为float, 长度为num x 128</returns>
        [DllImport(@"libGaius")]
        public static extern IntPtr Gaius_Forward(IntPtr instance, IntPtr faces, int num, int order);


    }
}
