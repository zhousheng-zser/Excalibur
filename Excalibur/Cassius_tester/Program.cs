using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using glasssix;

namespace Cassius_tester
{
    class Program
    {
        static void Main(string[] args)
        {
            unicorn uc = new unicorn(0);
            Bitmap bmp = new Bitmap("E:\\rec-bench\\uofw\\re_equalized\\Correct\\0\\0.jpg");
            float[] feature = uc.ExtractBitmapOutputs(new[] {bmp}, 0);
            for (int i = 0; i < feature.Length; i++)
            {
                Console.WriteLine(feature[i]);
            }
            Console.ReadLine();
        }
    }
}
