using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace G6.Algorithm.Lib
{
    [StructLayout(LayoutKind.Sequential)]
    public struct FaceRect
    {
        public int x;
        public int y;
        public int width;
        public int height;
        public int neighbors;
        /// <summary>
        /// 得分
        /// </summary>
        public double confidence;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct FaceRectwithInfo
    {
        public int x;
        public int y;
        public int width;
        public int height;
        public int neighbors;
        /// <summary>
        /// 得分
        /// </summary>
        public double confidence;

        [MarshalAsAttribute(UnmanagedType.ByValArray, SizeConst = 10, ArraySubType = UnmanagedType.I1)]
        public float[] pts;
        public float yaw;
        public float pitch;
        public float roll;

    }
	
	[StructLayout(LayoutKind.Sequential)]
	public struct Match_Retval {
		public FaceRect rect;
		
		[MarshalAs(UnmanagedType.ByValArray, SizeConst = 37)]
		public char[] id;
		
		[MarshalAs(UnmanagedType.I1)]
		public bool is_new;
	}

    public enum LonginusDetectionType
    {
        FRONTALVIEW = 0,
        FRONTALVIEW_REINFORCE = 1,
        MULTIVIEW = 2,
        MULTIVIEW_REINFORCE = 3
    }
    public class lib_longinucia
    {
        /// <summary>
        /// 获取实例
        /// </summary>
        /// <param name="device">使用的GPU索引，-1表示使用CPU</param>
        /// <returns>实例的指针</returns>
        [DllImport(@"libLonginus")]
        public static extern IntPtr Longinus_NewInstance(int device);
        /// <summary>
        /// 释放实例
        /// </summary>
        /// <param name="instance">实例的指针</param>
        [DllImport(@"libLonginus")]
        public static extern void Longinus_ReleaseInstance(IntPtr instance);
        /// <summary>
        /// 设置检测参数
        /// </summary>
        /// <param name="instance">实例的指针</param>
        /// <param name="type">检测函数，LonginusDetectionType所在的枚举</param>
        /// <param name="device">使用的GPU索引，-1表示使用CPU</param>
        [DllImport(@"libLonginus")]
        public static extern void Longinus_set(IntPtr instance, LonginusDetectionType type, int device);

        /// <summary>
        /// 人脸检测
        /// </summary>
        /// <param name="instance">实例的指针</param>
        /// <param name="rect_ptr">一个IntPtr变量的地址, 接收输出的检测结果, 此变量指向了FaceRect数组, 非托管内存需释放</param>
        /// <param name="gray">源图的灰度图</param>
        /// <param name="width">源图的宽</param>
        /// <param name="height">源图的高</param>
        /// <param name="step">图像每一行字节数，如果是灰度图就是等于宽度，如果是RGB就是等图宽乘以3</param>
        /// <param name="minSize">最小检测窗，最小值为24</param>
        /// <param name="scale">放大系数，一般为1.20f</param>
        /// <param name="min_neighbors">最小阈值通过阈值，一般为3</param>
        /// <returns>检测到的人脸个数</returns>

        [DllImport(@"libLonginus")]
        public static extern int Longinus_detect(IntPtr instance, out IntPtr rect_ptr, byte[] gray, int width, int height, int step, int minSize, float scale, int min_neighbors);

        /// <summary>
        /// 人脸检测,带关键点与角度检测
        /// </summary>
        /// <param name="instance">实例的指针</param>
        /// <param name="rect_ptr">一个IntPtr变量的地址, 接收输出的检测结果, 此变量指向了FaceRectwithInfo数组, 非托管内存需释放</param>
        /// <param name="gray">源图的灰度图</param>
        /// <param name="width">源图的宽</param>
        /// <param name="height">源图的高</param>
        /// <param name="step">图像每一行字节数，如果是灰度图就是等于宽度，如果是RGB就是等图宽乘以3</param>
        /// <param name="minSize">最小检测窗，最小值为24</param>
        /// <param name="scale">放大系数，一般为1.20f</param>
        /// <param name="min_neighbors">最小阈值通过阈值，一般为3</param>
        /// <param name="order">数据排列格式(NHWC:1 ,MCHW:0), 对于灰度图来说0和1一样</param>
        /// <returns>检测到的人脸个数</returns>
        [DllImport(@"libLonginus")]
        public static extern int Longinus_detectWithInfo(IntPtr instance, out IntPtr rect_ptr, byte[] gray, int width, int height, int step, int minSize, float scale, int min_neighbors, int order);

		/// <summary>
        /// 人脸追踪
        /// </summary>
        /// <param name="instance">实例的指针</param>
        /// <param name="match_retval_ptr">一个IntPtr变量的地址, 接收输出的追踪结果, 此变量指向了Match_Retval数组, 非托管内存需释放</param>
        /// <param name="rects">FaceRect数组的首地址</param>
        /// <param name="rect_num">FaceRect数组元素个数</param>
        /// <param name="frame_extract_frequency">帧提取频率</param>
        /// <param name="distance_factor">距离因子, 默认1.0f</param>
        /// <returns>Match_Retval数组元素个数</returns>
        [DllImport(@"libLonginus")]
        public static extern int Longinus_match(IntPtr instance, out IntPtr match_retval_ptr, IntPtr rects, int rect_num, int frame_extract_frequency, float distance_factor);

		/// <summary>
        /// 人脸追踪
        /// </summary>
        /// <param name="instance">实例的指针</param>
        /// <param name="match_retval_ptr">一个IntPtr变量的地址, 接收输出的追踪结果, 此变量指向了Match_Retval数组, 非托管内存需释放</param>
        /// <param name="rects">FaceRectwithInfo数组的首地址</param>
        /// <param name="rect_num">FaceRectwithInfo数组元素个数</param>
        /// <param name="frame_extract_frequency">帧提取频率</param>
        /// <param name="distance_factor">距离因子, 默认1.0f</param>
        /// <returns>Match_Retval数组元素个数</returns>
        [DllImport(@"libLonginus")]
        public static extern int Longinus_matchWithInfo(IntPtr instance, out IntPtr match_retval_ptr, IntPtr rects, int rect_num, int frame_extract_frequency, float distance_factor);

        /// <summary>
        /// 人脸检测(MTCNN)
        /// </summary>
        /// <param name="instance"></param>
        /// <param name="rect_ptr">一个IntPtr变量的地址, 接收输出的检测结果, 此变量指向了FaceRectwithInfo数组, 非托管内存需释放</param>
        /// <param name="image">只能是灰度图</param>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <param name="minSize">最小检测窗20</param>
        /// <param name="threshold">数组指针地址，一般为[0.6,0.7,0.7]</param>
        /// <param name="factor">缩放系数，1.41</param>
        /// <param name="stage">模型级数，一般为3</param>
        /// <param name="order">数据排列格式(NHWC:1 ,MCHW:0)</param>
        /// <returns>检测到的人脸个数</returns>
        [DllImport(@"libLonginus")]
        public static extern int Longinus_detectEx(IntPtr instance, out IntPtr rect_ptr, byte[] image, int height, int width, int minSize, float[] threshold, float factor, int stage, int order);

        /// <summary>
        /// 人脸检测(新版)
        /// </summary>
        /// <param name="instance"></param>
        /// <param name="rect_ptr">一个IntPtr变量的地址, 接收输出的检测结果, 此变量指向了FaceRectwithInfo数组, 非托管内存需释放</param>
        /// <param name="image">只能是灰度图</param>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <param name="order">数据排列格式(NHWC:1 ,MCHW:0)</param>
        /// <param name="threshold">门限，取0.5</param>
        /// <returns>检测到的人脸个数</returns>
        [DllImport(@"libLonginus")]
        public static extern int Longinus_detectRetina(IntPtr instance, out IntPtr rect_ptr, byte[] image, int height, int width, int order, int threshold);

        /// <summary>
        /// MTCNN，lite版本
        /// </summary>
        /// <param name="instance"></param>
        /// <param name="rect_ptr">一个IntPtr变量的地址, 接收输出的检测结果, 此变量指向了FaceRectwithInfo数组, 非托管内存需释放</param>
        /// <param name="image">只能是rgb</param>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <param name="minSize">最小检测窗20</param>
        /// <param name="threshold">数组指针地址，一般为[0.5,0.6,0.95]</param>
        /// <param name="factor">缩放系数，1.41</param>
        /// <param name="stage">模型级数，一般为3</param>
        /// <param name="order">数据排列格式</param>
        /// <returns>检测到的人脸个数</returns>
        [DllImport(@"libLonginus")]
        public static extern int Longinus_detectEx_Mobile(IntPtr instance, out IntPtr rect_ptr, byte[] image, int height, int width, int minSize, float[] threshold, float factor, int stage, int order);
        
		/// <summary>
        /// 红外检测，lite版本
        /// </summary>
        /// <param name="instance"></param>
        /// <param name="rect_ptr">一个IntPtr变量的地址, 接收输出的检测结果, 此变量指向了FaceRectwithInfo数组, 非托管内存需释放</param>
        /// <param name="image">只能是rgb</param>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <param name="minSize">最小检测窗20</param>
        /// <param name="threshold">数组指针地址，一般为[0.5,0.6,0.95]</param>
        /// <param name="factor">缩放系数，1.41</param>
        /// <param name="stage">模型级数，一般为3</param>
        /// <param name="order">数据排列格式</param>
		/// <returns>检测到的人脸个数</returns>
        [DllImport(@"libLonginus")]
        public static extern int Longinus_detectEx_Mobile_nir(IntPtr instance, out IntPtr rect_ptr, byte[] image, int height, int width, int minSize, float[] threshold, float factor, int stage, int order);
		
		/// <summary>
        /// 可见光图像和红外图像同时检测
        /// </summary>
        /// <param name="instance"></param>
        /// <param name="vsl_rect_ptr">一个IntPtr变量的地址, 接收输出的可见光图像检测结果, 此变量指向了FaceRectwithInfo数组, 非托管内存需释放</param>
        /// <param name="vsl_rect_num">一个int变量的地址, 接收输出的可见光图像检测人脸个数</param>
        /// <param name="vsl_image">可见光图片, 只能是rgb</param>
        /// <param name="vsl_height">可见光图片宽度</param>
        /// <param name="vsl_width">可见光图片高度</param>
        /// <param name="vsl_minSize">可见光图片最小检测窗20</param>
        /// <param name="vsl_threshold">可见光图片检测阈值, 数组指针地址，一般为[0.6,0.7,0.7]</param>
        /// <param name="vsl_factor">可见光图片缩放系数，1.41</param>
        /// <param name="vsl_stage">可见光图片模型级数，一般为3</param>
        /// <param name="vsl_order">可见光图片数据排列格式</param>
        /// <param name="nir_rect_ptr">一个IntPtr变量的地址, 接收输出的红外图像检测结果, 此变量指向了FaceRectwithInfo数组, 非托管内存需释放</param>
        /// <param name="nir_rect_num">一个int变量的地址, 接收输出的红外图像检测人脸个数</param>
        /// <param name="nir_image">红外图片, 只能是rgb</param>
        /// <param name="nir_height">红外图片宽度</param>
        /// <param name="nir_width">红外图片高度</param>
        /// <param name="nir_minSize">红外图片最小检测窗20</param>
        /// <param name="nir_threshold">红外图片检测阈值, 数组指针地址，一般为[0.5,0.6,0.95]</param>
        /// <param name="nir_factor">红外图片缩放系数，1.41</param>
        /// <param name="nir_stage">红外图片模型级数，一般为3</param>
        /// <param name="nir_order">红外图片数据排列格式</param>
		/// <returns></returns>
        [DllImport(@"libLonginus")]
        public static extern void detectEx_mobile_pair(IntPtr instance, out IntPtr vsl_rect_ptr, out int vsl_rect_num, byte[] vsl_image, int vsl_height, int vsl_width, int vsl_minSize, float[] vsl_threshold, float vsl_factor, int vsl_stage, int vsl_order,
														out IntPtr nir_rect_ptr, out int nir_rect_num, byte[] nir_image, int nir_height, int nir_width, int nir_minSize, float[] nir_threshold, float nir_factor, int nir_stage, int nir_order);


        /// <summary>
        /// 人脸对齐
        /// </summary>
        /// <param name="instance"></param>
		/// <param name="rect_ptr">一个IntPtr变量的地址, 接收输出的检测结果, 此变量指向了FaceRectwithInfo数组, 非托管内存需释放</param>
        /// <param name="gray">原始帧图片的灰度图</param>
        /// <param name="n">待对齐的人脸数</param>
        /// <param name="height">原始图片的高</param>
        /// <param name="width">原始图片的宽</param>
        /// <param name="bbox">人脸的rect拼接为一维数组([x1,y1,w1,h1,x2,y2,w2,h2....])</param>
        /// <param name="landmarks">关键点拼接为一维数组</param>
        /// <returns>对齐人脸数据首地址, 数据长度为nx3x128x128, 以NCHW排列, 非托管内存需释放</returns>

        [DllImport(@"libLonginus")]
        public static extern IntPtr Longinus_alignFace(IntPtr instance, byte[] gray, int n, int height, int width, int[] bbox, int[] landmarks);

        /// <summary>
        /// 人脸对齐
        /// </summary>
        /// <param name="instance"></param>
        /// <param name="gray">人脸放大40%裁剪后的灰度图, 以NCHW排列</param>
        /// <param name="n">人脸数, 目前仅能为1</param>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <returns>对齐人脸数据首地址, 数据类型为byte, 数据长度为nx3x128x128, 以NCHW排列, 非托管内存需释放</returns>

        [DllImport(@"libLonginus")]
        public static extern IntPtr Longinus_alignFaceFromCropped(IntPtr instance, byte[] gray, int n, int height, int width);



        /// <summary>
        /// 可见光模糊判断
        /// </summary>
        /// <param name="instance"></param>
        /// <param name="img">RGB图</param>
        /// <param name="height">原始RGB图的高</param>
        /// <param name="width">原始RGB图的宽</param>
        /// <param name="bbox"></param>
        /// <param name="landmark"></param>
        /// <param name="thresh">数组元素个数2个, 值要问张继</param>
        /// <param name="realthresh">接收实际阈值, 数组元素个数2个, 非托管内存需释放</param>
        /// <param name="order">数据排列格式</param>
        /// <returns>问张继</returns>
        [DllImport(@"libLonginus")]
        public static extern bool Longinus_blur_judge_vsl(IntPtr instance, byte[] img, int height, int width, int[] bbox, int[] landmark, float[] thresh, out IntPtr realthresh, int order);


        /// <summary>
        /// 可见光黑白模糊判断
        /// </summary>
        /// <param name="instance"></param>
        /// <param name="img">RGB图</param>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <param name="bbox"></param>
        /// <param name="landmark"></param>
        /// <param name="thresh">数组元素个数2个, 值要问张继</param>
        /// <param name="realthresh">接收实际阈值, 数组元素个数2个, 非托管内存需释放</param>
        /// <param name="order">数据排列格式</param>
        /// <returns>问张继</returns>

        [DllImport(@"libLonginus")]
        public static extern bool Longinus_black_white_judge_vsl(IntPtr instance, byte[] img, int height, int width, int[] bbox, int[] landmark, float[] thresh, out IntPtr realthresh, int order);


        /// <summary>
        /// 红外彩色照片判断
        /// </summary>
        /// <param name="instance"></param>
        /// <param name="img"></param>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <param name="bbox"></param>
        /// <param name="landmark"></param>
        /// <param name="thresh">数组元素个数2个, 值要问张继</param>
        /// <param name="realthresh">接收实际阈值, 数组元素个数2个, 非托管内存需释放</param>
        /// <param name="order"></param>
        /// <returns>问张继</returns>
        [DllImport(@"libLonginus")]
        public static extern bool Longinus_face_nose_judget_nir(IntPtr instance, byte[] img, int height, int width, int[] bbox, int[] landmark, float[] thresh, out IntPtr realthresh, int order);

        /// <summary>
        /// 获取版本信息
        /// </summary>
        /// <returns>版本字符串首地址, 非托管内存需释放</returns>
        [DllImport(@"libLonginus")]
        public static extern IntPtr Longinus_getVersion();
    }
}
