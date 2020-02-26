using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace G6.Algorithm.Lib
{
    [StructLayout(LayoutKind.Sequential)]
    public struct knn_mapping_data
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 128)]
        public float[] feature;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 33)]
        public char[] key;
        [MarshalAs(UnmanagedType.I1)]
        public bool is_active;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct knn_search_result
    {
        public knn_mapping_data data;
        public float distance_in_percentage;
    }
    public class lib_irisvian
    {
        /// <summary>
        /// 获取实例
        /// </summary>
        /// <param name="max_items">单个mapping文件最大存放数量</param>
        /// <param name="new_save_path">mapping文件存储目录, 文件目录需手动创建</param>
        /// <param name="tmp_path">build缓存文件存储目录, 文件目录需手动创建</param>
        /// <returns>实例的指针</returns>
        [DllImport(@"libIrisviel")]
        public static extern IntPtr Irisviel_NewInstance(int max_items, string new_save_path, string tmp_path);

        /// <summary>
        /// 释放实例
        /// </summary>
        /// <param name="instance">实例的指针</param>
        [DllImport(@"libIrisviel")]
        public static extern void Irisviel_ReleaseInstance(IntPtr instance);
        /// <summary>
        /// 获取maping文件存储目录
        /// </summary>
        /// <param name="instance"></param>
        /// <returns>mapping文件存储目录字符串首地址, 非托管内存需释放</returns>
        [DllImport(@"libIrisviel")]
        public static extern IntPtr Irisviel_save_path(IntPtr instance);
        /// <summary>
        /// 获取build缓存文件存储目录
        /// </summary>
        /// <param name="instance"></param>
        /// <returns>build缓存文件存储目录字符串首地址, 非托管内存需释放</returns>
        [DllImport(@"libIrisviel")]
        public static extern IntPtr Irisviel_tmp_path(IntPtr instance);

        /// <summary>
        /// build（构建人脸搜索库）
        /// </summary>
        /// <param name="instance"></param>
        /// <param name="file_count">mapping文件个数</param>
        /// <param name="ptr">ptr指向多个mapping文件路径字符串首地址数组的首地址。</param>
        /// <returns></returns>
        [DllImport(@"libIrisviel", CharSet = CharSet.Ansi)]
        //public static extern void Irisviel_build(IntPtr instance, int file_count, string[] filepath);
        public static extern void Irisviel_build(IntPtr instance, int file_count, IntPtr ptr);

        /// <summary>
        /// 搜索人员信息
        /// </summary>
        /// <param name="instance"></param>
        /// <param name="ptr_result">一个IntPtr变量的地址, 接收输出的搜索结果, 此变量指向了knn_search_result数组的首地址, 非托管内存需释放</param>
        /// <param name="feature">128维特征向量</param>
        /// <param name="top">指示输出最相似的前top个项目</param>
        /// <returns>搜索到的项目个数</returns>
        [DllImport(@"libIrisviel")]
        public static extern int Irisviel_search(IntPtr instance, out IntPtr ptr_result, float[] feature, int top);

        /// <summary>
        /// 清空整个数据库
        /// </summary>
        /// <param name="instance"></param>
        [DllImport(@"libIrisviel", CharSet = CharSet.Ansi)]
        public static extern void Irisviel_remove_all(IntPtr instance);


        /// <summary>
        /// 批量删除keys对应的人员信息
        /// </summary>
        /// <param name="instance"></param>
        /// <param name="ptr_count">一个int变量的地址, 此变量接收输出的删除文件数量</param>
        /// <param name="ptr_files">一个IntPtr变量的地址, 接收输出的删除文件路径，此变量指向多个文件路径字符串首地址数组的首地址, 非托管内存需释放</param>
        /// <param name="del_count">待删除数量</param>
        /// <param name="ptr_keys">待删除的key集合</param>
        [DllImport(@"libIrisviel", CharSet = CharSet.Ansi)]
        //public static extern void Irisviel_delete_features(IntPtr instance, out int ptr_count, out IntPtr ptr_files, int del_count, IntPtr ptr_keys);
        public static extern void Irisviel_delete_features(IntPtr instance, out int ptr_count, out IntPtr ptr_files, int del_count, string[] kyes);

        /// <summary>
        /// 删除key所对应的人员信息
        /// </summary>
        /// <param name="instance"></param>
        /// <param name="ptr_count">一个int变量的地址, 此变量接收输出的删除文件数量</param>
        /// <param name="ptr_files">一个IntPtr变量的地址, 接收输出的删除文件路径，此变量指向多个文件路径字符串首地址数组的首地址, 非托管内存需释放</param>
        /// <param name="key">待删除的key</param>
        [DllImport(@"libIrisviel", CharSet = CharSet.Ansi)]
        public static extern void Irisviel_delete_feature(IntPtr instance, out int ptr_count, out IntPtr ptr_files, string key);

        /// <summary>
        /// 批量添加人员信息
        /// </summary>
        /// <param name="instance"></param>
        /// <param name="count">待添加的特征个数</param>
        /// <param name="ptr_data">ptr_data指向了待添加的knn_mapping_data数组的首地址</param>
        [DllImport(@"libIrisviel", CharSet = CharSet.Ansi)]
        public static extern void Irisviel_add_features(IntPtr instance, int count, IntPtr ptr_data);


        /// <summary>
        /// 添加一个人员信息
        /// </summary>
        /// <param name="instance"></param>
        /// <param name="ptr_data">一个knn_mapping_data变量的引用</param>
        [DllImport(@"libIrisviel", CharSet = CharSet.Ansi)]
        public static extern void Irisviel_add_feature(IntPtr instance, ref knn_mapping_data ptr_data);


        /// <summary>
        /// 更新一个人员信息
        /// </summary>
        /// <param name="instance"></param>
        /// <param name="ptr_data">一个knn_mapping_data变量的引用</param>
        [DllImport(@"libIrisviel", CharSet = CharSet.Ansi)]
        public static extern void Irisviel_update_feature(IntPtr instance, ref knn_mapping_data ptr_data);
        /// <summary>
        /// 批量更新人员信息
        /// </summary>
        /// <param name="instance"></param>
        /// <param name="count">待添加的特征个数</param>
        /// <param name="ptr_data">ptr_data指向了待添加的knn_mapping_data数组的首地址</param>
        [DllImport(@"libIrisviel", CharSet = CharSet.Ansi)]
        public static extern void Irisviel_update_more(IntPtr instance, int count, IntPtr ptr_data);
    }
}
