using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using glasssix.longinus;

namespace CSharpExample
{
    class Program
    {
        static void Main(string[] args)
        {
            Bitmap bmp = new Bitmap(@"D:\Research\Excalibur\images\exciting.png");
            Longinucia longinucia = new Longinucia();
            longinucia.set(DetectorType.MULTIVIEW_REINFORCE, 0);
            var res = longinucia.Face_Detect(bmp, 24, 1.1f, 3, false, false, true);
            //var res_ex = longinucia.Face_DetectEx(bmp, 64, 1.414f, new float[3]{ 0.7f, 0.6f, 0.6f}, 3);
            for (int i = 0; i < res.Count; i++)
            {
                DrawRectangleInPicture(bmp,
                    new Rectangle(res[i].rect.X, res[i].rect.Y, res[i].rect.Width, res[i].rect.Height), Color.Azure, 2,
                    DashStyle.Dash);
            }
            //for (int i = 0; i < res_ex.Count; i++)
            //{
            //    DrawRectangleInPicture(bmp,
            //        new Rectangle(res_ex[i].rect.X, res_ex[i].rect.Y, res_ex[i].rect.Width, res_ex[i].rect.Height), Color.Crimson, 2,
            //        DashStyle.Dash);
            //}
            bmp.Save(@"D:\Research\Excalibur\images\exciting_res.png");
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
