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
//using glasssix.aroundight.romancia;
using glasssix.excalibur.cassius;
using glasssix.excalibur.longinus;

namespace Cassius_tester
{
    class Program
    {
        private static Bitmap bmp1;
        private static Bitmap bmp2;
        static void Main(string[] args)
        {
            Unicorn uc = new Unicorn(0);
            bmp1 = new Bitmap(@"C:\Users\BALTHASAR\Desktop\aligned.jpg");
            var aaa = uc.ExtractBitmapOutputs(new[] {bmp1});
            Stopwatch sw = new Stopwatch();
            sw.Start();
            for (int i = 0; i < 100; i++)
            {
                uc.ExtractBitmapOutputs(new[] { bmp1, bmp1 });
            }
            sw.Stop();
            Console.WriteLine(sw.ElapsedMilliseconds / 1);
            Console.ReadLine();
        }

        static void detection_test()
        {
            //FastFace mtcnn = new FastFace(1);
            //bmp1 = new Bitmap(@"E:\Data\LS3D-W\300W-Testset-3D\outdoor_236.png");
            ////var aaa = mtcnn.Facedetect_Multiview_CNN(bmp1, 40, 1.0f);
            ////var bbb = mtcnn.Facedetect_Multiview_Reinforce(bmp1, 40, 1.2f);

            //Stopwatch sw = new Stopwatch();
            //sw.Start();
            //var auto = mtcnn.Facedetect_Multiview_CNN(bmp1, 48, 1.2f);
            //Console.WriteLine(auto[0].score);
            //for (int i = 0; i < 5; i++)
            //{
            //    mtcnn.Facedetect_Multiview_CNN(bmp1, 48, 1.2f);
            //}
            //sw.Stop();
            //Console.WriteLine(sw.ElapsedMilliseconds / 5);
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
