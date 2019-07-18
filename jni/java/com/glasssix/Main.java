package com.glasssix;
import com.glasssix.Gaiulinya.Gaiulinya;
import com.glasssix.Cassiutia.Cassiutia;
import com.glasssix.Irisvika.Irisvika;
import com.glasssix.Longimila.DetectionType;
import com.glasssix.Longimila.FaceRect;
import com.glasssix.Longimila.FaceRectwithFaceInfo;
import com.glasssix.Longimila.Longimila;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Main {

    public static void main(String[] args) {
        System.loadLibrary("opencv_java400");
        System.out.println(Gaiulinya.getVersion());
        String v = Cassiutia.getVersion();
        System.out.println(v);
        Irisvika s = new Irisvika(512);
        Longimila ll = new Longimila(-1);
        Gaiulinya gg = new Gaiulinya(-1);
        ll.set(DetectionType.FRONTALVIEW, -1);
        Mat img = Imgcodecs.imread("C:\\Users\\Glasssix-Admin\\Desktop\\exciting.png");
        Mat gray = new Mat();
        Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);
        FaceRect[] re = ll.detect(gray.getNativeObjAddr(),100, 1.2F, 3);
        FaceRectwithFaceInfo[] rein = ll.detectwithInfo(gray.getNativeObjAddr(), 100, 1.2f, 3, 1);
        //if (re.length>0)
        //{
        //    Imgproc.rectangle(img, new Rect(re[0].x, re[0].y,re[0].width,re[0].height), new Scalar(0,0,255));
        //}
        byte[] aligned_face_data = ll.alignFace(gray.getNativeObjAddr(), rein);
        Mat[] aligned_faces = encode2mats(aligned_face_data, rein.length);
        HighGui.imshow("test", aligned_faces[0]);
        HighGui.waitKey();
        float[][] features = gg.ForwardwithMetaData(aligned_face_data, rein.length, 0);
        System.out.println(features[0][127]);
    }

    public static Mat[] encode2mats(byte[] metadata, int face_count)
    {
        Mat[] faces = new Mat[face_count];
        if (metadata.length != face_count * 3 * 128 *128)
        {
            return faces;
        }
        int n_offset = 3 * 128 *128;
        int c_offset = 128 * 128;
        for (int i = 0; i < face_count; i++)
        {
            byte[] dst_data = new byte[3 *128 *128];
            faces[i] = new Mat(128, 128, CvType.CV_8UC3);
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < 128; h++)
                {
                    for (int w = 0; w < 128; w++)
                    {
                        dst_data[h * 3 * 128 + w * 3 + c]
                            = metadata[n_offset * i + c * c_offset + h * 128 + w];
                    }
                }
            }
            faces[i].put(0,0, dst_data);
        }
        return faces;
    }
}
