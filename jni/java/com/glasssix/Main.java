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

import java.awt.Graphics;
import java.awt.Image;
import java.awt.Label;
import java.awt.MediaTracker;
import java.awt.Toolkit;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.io.IOException;

import javax.imageio.ImageIO;

public class Main {

    public static void main(String[] args) {
        System.loadLibrary("opencv_java400");
        System.out.println(Gaiulinya.getVersion());
        String v = Cassiutia.getVersion();
        System.out.println(v);
        Longimila ll = new Longimila(-1);
        Gaiulinya gg = new Gaiulinya(-1);
        ll.set(DetectionType.FRONTALVIEW, -1);
        Mat img = Imgcodecs.imread("C:\\Users\\Glasssix-Admin\\Desktop\\exciting.png");
        Mat gray = new Mat();
        Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);
        FaceRect[] re = ll.detectbyMetaData(encode2bytes(gray), gray.width(), gray.height(), 100, 1.2F, 3);
        FaceRectwithFaceInfo[] rein = ll.detectwithInfobyMetaData(encode2bytes(gray), gray.width(), gray.height(), 100, 1.2f, 3, 1);
        //if (re.length>0)
        //{
        //    Imgproc.rectangle(img, new Rect(re[0].x, re[0].y,re[0].width,re[0].height), new Scalar(0,0,255));
        //    HighGui.imshow("test", img);
        //    HighGui.waitKey();
        //}
        byte[] aligned_face_data = ll.alignFacebyMetaData(encode2bytes(gray), gray.width(), gray.height(), rein);
        Mat[] aligned_faces = encode2mats(aligned_face_data, rein.length);
        if (aligned_faces.length>0)
        {
            HighGui.imshow("align", aligned_faces[0]);
            HighGui.waitKey();
        }
        Mat img2 = Imgcodecs.imread("C:\\Users\\Glasssix-Admin\\Desktop\\yswvisible.jpg");
        Mat img3 = Imgcodecs.imread("C:\\Users\\Glasssix-Admin\\Desktop\\yswinfread.jpg");
        float[][] features = gg.ForwardbyMetaData(aligned_face_data, rein.length, 0);
        float[][] feat2 = gg.ForwardbyMetaData(encode2bytes(img2), 1, 1);
        float[][] feat3 = gg.ForwardbyMetaData(encode2bytes(img3), 1, 1);
        Irisvika s = new Irisvika(128);
        String graph_path = "D:\\Research\\Excalibur\\data\\test.graph";
        String data_path = "D:\\Research\\Excalibur\\data\\test.data";
        s.buildGraphwithData(new float[][]{feat3[0], features[0]});
        //s.saveGraph("D:\\Research\\Excalibur\\data\\test.graph");

        //s.saveGraphwithData(graph_path, data_path);
        //s.loadGraph("D:\\Research\\Excalibur\\data\\test.graph");
        //s.loadGraphwithData(graph_path, data_path);
        s.optimizeGraph();
        int topK = 1;
        int [][] ids = new int[1][];
        float [][] similaries = new float[1][];
        for (int i = 0; i < topK; i++)
        {
            ids[i] = new int[topK];
            similaries[i] = new float[topK];
        }
        s.searchVector(feat2, 1, ids, similaries);
        System.out.println(features[0][127]);
    }

    public static byte[] encode2bytes(Mat img)
    {
        byte[] buffer = new byte[img.cols() * img.rows() * img.channels()];
        img.get(0,0, buffer);
        return buffer;
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
