package com.glasssix.longinus;

import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;


public class Point {
	public int x;
	public int y;
	
	public Point() {
	}
	
	public Point(int x, int y) {
		this.x = x;
		this.y = y;
	}
}

public class FaceRect {
	public int x;
	public int y;
	public int width;
	public int height;
	public int neighbors;
	public double confidence;
	
	public FaceRect() {
		x = 0;
		y = 0;
		width = 0;
		height = 0;
		neighbors = 0;
		confidence = 0.0;
	}
	
	public FaceRect(int x, int y, int width, int height, int neighbors, double confidence) {
		this.x = x;
		this.y = y;
		this.width = width;
		this.height = height;
		this.neighbors = neighbors;
		this.confidence = confidence;
	}
}

public class FaceRectwithFaceInfo extends FaceRect {
	public Point pts[5];
	public float yaw;
	public float pitch;
	public float roll;
	
	public FaceRectwithFaceInfo() {
	}
	
	public FaceRectwithFaceInfo(FaceRect rect) {
		super.rect = rect;
	}
}

public class Match_Retval {
	public FaceRect rect;
	public String id;
	public boolean is_new;
	
	public Match_Retval() {
	}
	
	public Match_Rectval(FaceRect rect, String id, boolean is_new) {
		this.rect = rect;
		this.id = id;
		this.is_new = is_new;
	}
}

public enum DetectionType {
	FRONTALVIEW, FRONTALVIEW_REINFORCE, MULTIVIEW, MULTIVIEW_REINFORCE
}

public class Longinus {
    static {
		System.loadLibrary("Longinus-java")
	}
	
	private long mObject;
	
	public Longinus(int device) {
		init(device);
	}
	
	public void set(DetectionType detectionType, int device) {
		switch(detectionType) {
		case FRONTALVIEW:
			set(0, device);
			break;
		case FRONTALVIEW_REINFORCE:
			set(1, device);
			break;
		case MULTIVIEW:
			set(2, device);
			break;
		case MULTIVIEW_REINFORCE:
			set(3, device);
			break;
		}
	}
	
	public static native String getVersion();
	public native FaceRect[] detect(long grayNativeObj, int minSize, float scale, int minNeighbors);
	public native FaceRectwithFaceInfo[] detect(long grayNativeObj, int minSize, float scale, int minNeighbors, int order);
	public native Match_Retval[] match(FaceRect[] faceRect, int frame_extract_frequency);
	public native Byte[] alignFace(long grayNativeObj, int[] bbox, int[] landmarks);
	public native Byte[] alignFace(long grayNativeObj);
	private native void init(int device);
	private native void set(int detectionType, int device);
	protected native void finalize();
}