using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using glasssix.longinus;
using glasssix.cassius;
using System.Diagnostics;

namespace TestLonginucia
{
    class Program
    {
        static void Main(string[] args)
        {
            testlonginus();
            //Cassiunia cc = new Cassiunia(-1);
            //Bitmap bmp = new Bitmap(@"D:\yswinfread.jpg");
            //Bitmap bmp2 = new Bitmap(@"D:\yswvisible.jpg");
            //var res1 = cc.ExtractBitmapOutputs(new Bitmap[] { bmp });
            //var res2 = cc.ExtractBitmapOutputs(new Bitmap[] { bmp2 });
            //Stopwatch sw = new Stopwatch();
            //sw.Start();
            //for (int i = 0; i < 1000; i++)
            //{
            //    cc.ExtractBitmapOutputs(new Bitmap[] { bmp2 });
            //}
            //sw.Stop();
            //Console.WriteLine("Cost " + sw.ElapsedMilliseconds / 1000 + "ms");



            //for (int i = 0; i < 1; i++)
            //{
            //    for (int j = 0; j < 50; j++)
            //    {
            //        Console.Write(res1[i * 512 + j] + " ");
            //    }
            //    Console.WriteLine();
            //}
            //Console.WriteLine(Cassiunia.CosineDistanceProb(res1, res2));
            Console.ReadLine();
        }

        static void testlonginus()
        {
            Bitmap bmp = new Bitmap(@"D:\projects\MCL_Forward\Excalibur\TestImages\xiaoyuankeji.jpg");
            //Bitmap bmp = new Bitmap(@"D:\a.png");
            Longinucia gg = new Longinucia(bmp.Width, bmp.Height);
            gg.set(glasssix.longinus.DetectorType.FRONTALVIEW_REINFORCE, -1);
            var res = gg.Face_Detect(bmp, 60, 1.1f, 3, false, false, true);
            System.Diagnostics.Stopwatch sw = new Stopwatch();
            sw.Start();
            for (int i = 0; i < 0; i++)
            {
                gg.Face_Detect(bmp, 24, 1.1f, 3, false, false, true);
            }
            sw.Stop();
            Console.WriteLine("Cost " + sw.ElapsedMilliseconds / 100 + "ms");
            //Console.ReadLine();
            //for (int i = 0; i < res.Count; i++)
            //{
            //    DrawRectangleInPicture(bmp, new Point(res[i].rect.X, res[i].rect.Y),
            //        new Point(res[i].rect.X + res[i].rect.Width, res[i].rect.Y + res[i].rect.Height),
            //        Color.Azure, 2, System.Drawing.Drawing2D.DashStyle.Solid);
            //    for (int j = 0; j < 5; j++)
            //    {
            //        DrawRoundInPicture(bmp, new Point(res[i].landmarks[2 * j + 0] - 1, res[i].landmarks[2 * j + 1] - 1),
            //            new Point(res[i].landmarks[2 * j + 0] + 1, res[i].landmarks[2 * j + 1] + 1),
            //            Color.Crimson, 2, System.Drawing.Drawing2D.DashStyle.Solid);
            //    }
            //}
            //gg.Match_Faces(ref res, 1);
            var outputs = gg.AlignFace(res);
            outputs[2].Save(@"D:\projects\MCL_Forward\Excalibur\TestImages\res.jpg");
        }

        public static Bitmap DrawRectangleInPicture(Bitmap bmp, Point p0, Point p1, Color RectColor, int LineWidth, System.Drawing.Drawing2D.DashStyle ds)
        {
            if (bmp == null) return null;


            Graphics g = Graphics.FromImage(bmp);

            Brush brush = new SolidBrush(RectColor);
            Pen pen = new Pen(brush, LineWidth);
            pen.DashStyle = ds;

            g.DrawRectangle(pen, new Rectangle(p0.X, p0.Y, Math.Abs(p0.X - p1.X), Math.Abs(p0.Y - p1.Y)));

            g.Dispose();

            return bmp;
        }

        public static Bitmap DrawRoundInPicture(Bitmap bmp, Point p0, Point p1, Color RectColor, int LineWidth, System.Drawing.Drawing2D.DashStyle ds)
        {
            if (bmp == null) return null;

            Graphics g = Graphics.FromImage(bmp);

            Brush brush = new SolidBrush(RectColor);
            Pen pen = new Pen(brush, LineWidth);
            pen.DashStyle = ds;

            g.DrawEllipse(pen, new Rectangle(p0.X, p0.Y, Math.Abs(p0.X - p1.X), Math.Abs(p0.Y - p1.Y)));

            g.Dispose();

            return bmp;
        }
    }
}
