using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using glasssix.longinus;
using glasssix.cassius;
using glasssix.gaius;
using glasssix.excalibur;

namespace CSharpExample
{
    class Program
    {
        static void Main(string[] args)
        {
            int device = -1;
            Bitmap bmp2 = new Bitmap(@"D:\rr.jpg");
            Bitmap bmp = new Bitmap(@"D:\xiaoyuankeji.jpg");
            Longinucia longinucia = new Longinucia();
            longinucia.set(DetectorType.MULTIVIEW_REINFORCE, device);
            var res = longinucia.Face_Detect(bmp, 24, 1.1f, 3, false, false, true);
            TensorBuilder b = new TensorBuilder();
            //var res = longinucia.Face_DetectEx(bmp, 64, 1.414f, new float[3]{ 0.7f, 0.6f, 0.6f}, 3);
            //for (int i = 0; i < 100; i++)
            //{
            //    longinucia.Face_DetectEx(bmp, 64, 1.414f, new float[3] { 0.7f, 0.6f, 0.6f }, 3);
            //}
            var aligned_faces = longinucia.AlignFace(bmp, res);
            for (int i = 0; i < aligned_faces.Length; i++)
            {
                aligned_faces[i].Save(@"D:\xiaoyuankeji_align"+i+".jpg");
            }

            var aligned_faces2 = longinucia.AlignFace(bmp2);
            if (device < 0)
            {
                aligned_faces2.Save(@"D:\rr_cpu_align.jpg");
            }
            else
            {
                aligned_faces2.Save(@"D:\rr_gpu_align.jpg");
            }

            //aligned_faces[0].Save(@"D:\xiaoyuankeji_align.jpg");
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
            bmp.Save(@"D:\720_res.jpg");

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
