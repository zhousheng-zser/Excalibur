using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using glasssix;
using glasssix.aroundight.romancia;

namespace Cassius_tester
{
    class Program
    {
        private static Bitmap bmp1;
        private static Bitmap bmp2;
        static void Main(string[] args)
        {
            FastCNNFace mtcnn = new FastCNNFace(0);
            bmp1 = new Bitmap("C:\\Users\\BALTHASAR\\Desktop\\WeChat Image_20180309174405.jpg");
            var aaa = mtcnn.Facedetect_Mtcnn(bmp1, 40);
            Stopwatch sw = new Stopwatch();
            sw.Start();
            for (int i = 0; i < 10; i++)
            {
                mtcnn.Facedetect_Mtcnn(bmp1, 40);
            }
            sw.Stop();
            Console.WriteLine(sw.ElapsedMilliseconds/10);
            //DrawRectangleInPicture(bmp1, aaa[0].rect, Color.Aquamarine, 2, DashStyle.DashDot);
            //bmp1.Save("C:\\Users\\BALTHASAR\\Desktop\\detected.jpg");
            Console.ReadLine();
        }

        //public static void multi_thread_tester1(object extractor)
        //{
        //    unicorn uc = extractor as unicorn;
        //    for (int i = 0; i < 100; i++)
        //    {
        //        uc.ExtractBitmapOutputs(new[] { bmp1 }, 0);
        //        Console.WriteLine("Thread 1: the "+ i +"-th execution.");
        //    }
        //}

        //public static void multi_thread_tester2(object extractor)
        //{
        //    unicorn uc = extractor as unicorn;
        //    for (int i = 0; i < 100; i++)
        //    {
        //        uc.ExtractBitmapOutputs(new[] { bmp2 }, 0);
        //        Console.WriteLine("Thread 2: the " + i + "-th execution.");
        //    }
        //}

        public static Bitmap DrawRectangleInPicture(Bitmap bmp, Rectangle rect, Color RectColor, int LineWidth, DashStyle ds)
        {
            if (bmp == null) return null;


            Graphics g = Graphics.FromImage(bmp);

            Brush brush = new SolidBrush(RectColor);
            Pen pen = new Pen(brush, LineWidth);
            pen.DashStyle = ds;

            g.DrawRectangle(pen, rect);

            g.Dispose();

            return bmp;
        }
    }
}
