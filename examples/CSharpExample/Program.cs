using System;
using System.IO;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using glasssix.longinus;
using glasssix.cassius;
using glasssix.gaius;
//using glasssix.athene;

namespace CSharpExample
{
    class Program
    {
        static void Main(string[] args)
        {
            int device = 0;
            Longinucia longinucia = new Longinucia();
            longinucia.set(DetectorType.MULTIVIEW_REINFORCE, device);

            int loop = 1;
            //Stopwatch stopwatch = new Stopwatch();
            //stopwatch.Start();

            Bitmap bmp = new Bitmap(@"D:/img/122331199110122212.bmp");
            //Bitmap bmp = new Bitmap(@"D:/640.bmp");

            var res = longinucia.Face_DetectEx(bmp, 48, 1.41f, new float[3] { 0.8f, 0.8f, 0.6f }, 3);
            //var res = longinucia.Face_DetectEx_mobile(bmp, 64, 1.414f, new float[3] { 0.7f, 0.6f, 0.6f }, 3);
            //for (int i = 0; i < loop; i++)
            //{
            //    res = longinucia.Face_DetectEx(bmp, 48, 1.41f, new float[3] { 0.8f, 0.8f, 0.6f }, 3);
            //}

            //stopwatch.Stop();
            //TimeSpan timespan = stopwatch.Elapsed;
            //double milliseconds = timespan.TotalMilliseconds / loop;
            //string out_txt = @"D:\csharp.txt";
            //FileStream fs = new FileStream(out_txt, FileMode.Truncate);
            //StreamWriter wr = null;
            //wr = new StreamWriter(fs);
            //wr.WriteLine(milliseconds);
            //wr.Close();

            var aligned_faces = longinucia.AlignFace(bmp, res);
            for (int i = 0; i < aligned_faces.Length; i++)
            {
                aligned_faces[i].Save(@"D:\align" + i + ".jpg");
            }

            for (int i = 0; i < res.Count; i++)
            {
                DrawRectangleInPicture(bmp,
                    new Rectangle(res[i].rect.X, res[i].rect.Y, res[i].rect.Width, res[i].rect.Height), Color.Azure, 2,
                    DashStyle.Dash);
            }

            bmp.Save(@"D:\detect_res.jpg");

            return;
        }

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


//namespace CSharpExample
//{
//    class Program
//    {
//        static void Main(string[] args)
//        {
//            Bitmap bmp = new Bitmap(@"D:/projects/test/poseprofiler/poseprofiler/poseimages/image_856.jpg");
//            int scale = 4;
//            int base_height = 72 * scale;
//            int base_width = 128 * scale;
//            string stream = "rtsp://admin:hk123456@192.168.1.64:554/h264/ch1/main/av_stream";
//            string deploy = "D:/projects/test/poseprofiler/poseprofiler/pose_deploy_linevec.prototxt";
//            string caffemodel = "D:/projects/test/poseprofiler/poseprofiler/pose_iter_440000.caffemodel";
//            int device = 0;
//            Athenel pose_profiler = new Athenel(stream, deploy, caffemodel, base_height, base_width, device);
//            //pose_profiler.Forward();
//            var res = pose_profiler.Forward(bmp);
//            res.Save(@"D:\res_856.jpg");
//        }
//    }
//}


//namespace CSharpExample
//{
//    class Program
//    {
//        static void Main(string[] args)
//        {
//            int device = -1;
//            Bitmap bmp = new Bitmap(@"D:\yswinfread.jpg");
//            Bitmap[] arr = { bmp };
//            Cassiunia Cassiuncia = new Cassiunia(device);
//            Gaiunia Gaiunicia = new Gaiunia(device);

//            for (int i = 0; i < 100000; i++)
//            {
//                var res512 = Cassiuncia.ExtractBitmapOutputs(arr);
//                var res128 = Gaiunicia.ExtractBitmapOutputs(arr);

//                double sum = 0;
//                for (int j = 0; j < 512; j++)
//                {
//                    sum += res512[j] * res512[j];
//                }

//                sum = Math.Sqrt(sum);
//            }

//        }
//    }
//}