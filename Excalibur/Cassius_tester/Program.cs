using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using glasssix;

namespace Cassius_tester
{
    class Program
    {
        private static Bitmap bmp1;
        private static Bitmap bmp2;
        static void Main(string[] args)
        {
            unicorn uc1 = new unicorn(0);
            unicorn uc2 = new unicorn(0);
            bmp1 = new Bitmap("E:\\rec-bench\\uofw\\re_equalized\\Correct\\0\\0.jpg");
            bmp2 = new Bitmap("E:\\rec-bench\\uofw\\re_equalized\\Correct\\0\\0.jpg");
            Thread t1 = new Thread(new ParameterizedThreadStart(multi_thread_tester1));
            Thread t2 = new Thread(new ParameterizedThreadStart(multi_thread_tester2));
            t1.IsBackground = true;
            t2.IsBackground = true;
            t1.Start(uc1);
            t2.Start(uc2);
            Console.ReadLine();
        }

        public static void multi_thread_tester1(object extractor)
        {
            unicorn uc = extractor as unicorn;
            for (int i = 0; i < 100; i++)
            {
                uc.ExtractBitmapOutputs(new[] { bmp1 }, 0);
                Console.WriteLine("Thread 1: the "+ i +"-th execution.");
            }
        }

        public static void multi_thread_tester2(object extractor)
        {
            unicorn uc = extractor as unicorn;
            for (int i = 0; i < 100; i++)
            {
                uc.ExtractBitmapOutputs(new[] { bmp2 }, 0);
                Console.WriteLine("Thread 2: the " + i + "-th execution.");
            }
        }
    }
}
