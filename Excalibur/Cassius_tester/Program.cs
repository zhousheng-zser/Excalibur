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
//using glasssix.excalibur.cassius;
using glasssix.excalibur.longinus;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using glasssix.gilgamesh;

namespace Cassius_tester
{
    class Program
    {
        private static Bitmap bmp1;
        private static Bitmap bmp2;
        static void Main(string[] args)
        {
            //unicorntest();
            Banshee be = new Banshee(0);
            //Unicorn uc = new Unicorn(0);
            bmp1 = new Bitmap(@"F:\bing\detected_img\0\152612277\1471065472304142_155.jpg");
            bmp2 = new Bitmap(@"F:\bing\detected_img\0\152612277\1471059678658332_47.jpg");
            Tensor a = new Tensor(bmp2, -1);
            Tensor b = new Tensor();
            tensorcv.resize(a, ref a, 500, 500, InterpolationType.Nearest, -1);
            a.Save(@"C:\Users\BALTHASAR\Desktop\00.png", ImageEncodingType.Png);
            float[][] ipbbox = be.ExtractBitmapOutputs_IPBbox(new[] { bmp1 });
            var ccc = Aligement(new[] { bmp1 }, ipbbox, be);
            var aaa = be.align(new[] { bmp1 });
            //Stopwatch sw = new Stopwatch();
            //float[] cs = uc.ExtractBitmapOutputs(new[] { ccc[0] });
            //float[] cpp = uc.ExtractBitmapOutputs(new[] { aaa[0].align_face });
            //float prob = Unicorn.CosineDistanceProb(cs, cpp);
            ccc[0].Save(@"C:\Users\BALTHASAR\Desktop\algnmenttest\cs.jpg");
            aaa[0].align_face.Save(@"C:\Users\BALTHASAR\Desktop\algnmenttest\cpp.jpg");
            //Console.WriteLine(prob);
            //sw.Start();
            //for (int i = 0; i < 1000; i++)
            //{
            //    be.align(new[] {bmp1});
            //}
            //sw.Stop();
            //Console.WriteLine(sw.ElapsedMilliseconds / 1);

            //Console.WriteLine(String.Format("Yaw: {0}, Pitch: {1}, Roll: {2}", aaa[0].yaw, aaa[0].pitch, aaa[0].roll));
            //Console.WriteLine(String.Format("Yaw: {0}, Pitch: {1}, Roll: {2}", aaa[1].yaw, aaa[1].pitch, aaa[1].roll));
            Console.ReadLine();
        }

        static void unicorntest()
        {
            //Unicorn uc = new Unicorn(0);
            //bmp1 = new Bitmap(@"C:\Users\BALTHASAR\Desktop\aligned2.jpg");
            //uc.ExtractBitmapOutputs(new[] { bmp1 });
            //Stopwatch sw = new Stopwatch();
            //sw.Start();
            //for (int i = 0; i < 100; i++)
            //{
            //    uc.ExtractBitmapOutputs(new[] { bmp1, bmp1, bmp1, bmp1, bmp1 });
            //}
            //sw.Stop();
            //float[] qs = uc.GetQualityScores();
            //Console.WriteLine(qs[0]);
            //Console.WriteLine(sw.ElapsedMilliseconds / 1);
        }

        public static Bitmap[] Aligement(Bitmap[] bits, float[][] ipbbox, glasssix.excalibur.longinus.Banshee aligementModule)
        {
            Bitmap[] bits_Result = new Bitmap[bits.Length];
            for (int j = 0; j < bits.Length; j++)
            {
                Image<Bgr, byte> image = new Image<Bgr, byte>(bits[j]);
                Rectangle draw = new Rectangle(Convert.ToInt32(ipbbox[0][j * 4 + 0] * image.Width),
                    Convert.ToInt32(ipbbox[0][j * 4 + 1] * image.Height),
                    Convert.ToInt32(ipbbox[0][j * 4 + 2] * image.Width - ipbbox[0][j * 4 + 0] * image.Width),
                    Convert.ToInt32(ipbbox[0][j * 4 + 3] * image.Height - ipbbox[0][j * 4 + 1] * image.Height));
                Mat mat = new Mat();
                PointF center = new PointF((ipbbox[0][j * 4 + 2] * image.Width + ipbbox[0][j * 4 + 0] * image.Width) / 2, (ipbbox[0][j * 4 + 3] * image.Height + ipbbox[0][j * 4 + 1] * image.Height) / 2);
                double angeltheta = ipbbox[1][j * 3 + 2] * 90;
                double arctheta = angeltheta * Math.PI / 180;
                CvInvoke.GetRotationMatrix2D(center, -1 * angeltheta, 1, mat);
                image = image.WarpAffine(mat, Inter.Cubic, Warp.FillOutliers, Emgu.CV.CvEnum.BorderType.Constant, new Bgr(Color.Black));
                int delta = Convert.ToInt32(Math.Sin(arctheta) * draw.Height / 2); //Correction 1
                draw.X += delta;
                center.X += delta;
                int delta_pitch =
                    Convert.ToInt32((1 - Math.Cos(Convert.ToDouble(ipbbox[1][j * 3 + 1] * 90 * Math.PI / 180))) * draw.Height / 2);
                //Correction 2
                draw.Y -= delta_pitch;
                draw.Height += 2 * delta_pitch;
                //image.Draw(draw, new Bgr(Color.Aqua), image.Width/64);
                //
                int Margin_Height = Convert.ToInt32(draw.Height * 1.2);
                int Margin_X = Convert.ToInt32(center.X - Margin_Height / 2);
                int Margin_Y = Convert.ToInt32(center.Y - Margin_Height / 2);
                Rectangle Margindraw = new Rectangle(Margin_X, Margin_Y, Margin_Height, Margin_Height);
                //What's you really need
                //image.Draw(Margindraw, new Bgr(Color.Crimson), image.Width / 64);
                image.ROI = Margindraw;
                Image<Bgr, byte> C = new Image<Bgr, byte>(image.Bitmap);
                C.Save(@"C:\Users\BALTHASAR\Desktop\algnmenttest\C.jpg");
                float[] pts5 = aligementModule.ExtractBitmapOutputs_IPTs(new[] { C.Bitmap });
                for (int k = 0; k < 5; k++)
                {
                    CircleF temp = new CircleF(new PointF(pts5[2 * k] * C.Width, pts5[2 * k + 1] * C.Height), 1);
                    C.Draw(temp, new Bgr(Color.Aqua), C.Width / 64);
                }

                PointF center_eye = new PointF((pts5[0] + pts5[2]) / 2 * C.Width, (pts5[1] + pts5[3]) / 2 * C.Height);
                PointF center_mouth = new PointF((pts5[6] + pts5[8]) / 2 * C.Width,
                    (pts5[7] + pts5[9]) / 2 * C.Height);
                CircleF center_eye_circle = new CircleF(center_eye, 1);
                C.Draw(center_eye_circle, new Bgr(Color.Crimson), C.Width / 64);
                //************* 待修改
                CircleF half_square = new CircleF(new PointF(0.5f * C.Width, 0.25f * C.Height), 1);
                C.Draw(half_square, new Bgr(Color.Chartreuse), C.Width / 64);

                float distance_x = center_eye.X - half_square.Center.X;
                float distance_y = center_eye.Y - half_square.Center.Y;

                float distance_me =
                    (float)
                    Math.Sqrt(Math.Pow((center_mouth.X - center_eye.X), 2) +
                              Math.Pow((center_mouth.Y - center_eye.Y), 2));

                float scale = distance_me / (C.Height * 0.5f);
                Margindraw.X += (int)distance_x;
                Margindraw.Y += (int)distance_y;

                image.ROI = Rectangle.Empty;
                image.ROI = Margindraw;

                //*****
                Mat mat_2 = new Mat();

                double tan = ((pts5[1] - pts5[3])) / (pts5[0] + pts5[2]);
                double arctan = Math.Atan(tan) * 180 / Math.PI;

                CvInvoke.GetRotationMatrix2D(
                    new PointF(half_square.Center.X + Margindraw.X, half_square.Center.Y + Margindraw.Y), -3 * arctan, 1,
                    mat_2);

                image.ROI = Rectangle.Empty;

                image = image.WarpAffine(mat_2, Inter.Cubic, Warp.FillOutliers, Emgu.CV.CvEnum.BorderType.Constant,
                    new Bgr(Color.Black));
                image.ROI = Margindraw;
                //************* 待修改
                Margindraw.X = Margindraw.X + Convert.ToInt32((1 - scale) * Margindraw.Width * 0.5f);
                Margindraw.Y = Margindraw.Y + Convert.ToInt32((1 - scale) * Margindraw.Height * 0.25f);
                /***********************/
                Margindraw.Width = Convert.ToInt32(Margindraw.Width * scale);
                Margindraw.Height = Convert.ToInt32(Margindraw.Height * scale);
                image.ROI = Rectangle.Empty;
                image.ROI = Margindraw;
                Image<Bgr, byte> F = new Image<Bgr, byte>(image.Bitmap);
                Rectangle final = new Rectangle(Convert.ToInt32(0.0f / 14 * F.Width), 0, Convert.ToInt32(14.0f / 14 * F.Width),
                    F.Height);
                Image<Bgr, byte> G = new Image<Bgr, byte>(F.Bitmap);
                G.ROI = final;
                Image<Gray, byte> gray = new Image<Gray, byte>(G.Size);
                CvInvoke.CvtColor(G, gray, ColorConversion.Bgr2Gray);
                gray._EqualizeHist();
                Image<Bgr, byte> colorimg = new Image<Bgr, byte>(new Image<Gray, byte>[] { gray, gray, gray });
                colorimg = colorimg.Resize(128, 128, Inter.Cubic);
                bits_Result[j] = new Bitmap(colorimg.Bitmap);
                mat.Dispose();
                mat_2.Dispose();
                gray.Dispose();
                C.Dispose();
                G.Dispose();
                F.Dispose();
                image.Dispose();
            }
            return bits_Result;
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
