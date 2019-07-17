package com.glasssix.Longimila;

public class Longimila {
    static {
		System.loadLibrary("Longimila");
	}
	
	private long mObject;
	
	public Longimila(int device) {
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
	public native Match_Rectval[] match(FaceRect[] faceRect, int frame_extract_frequency);
	public native Byte[] alignFace(long grayNativeObj, int[] bbox, int[] landmarks);
	public native Byte[] alignFace(long grayNativeObj);
	private native void init(int device);
	private native void set(int detectionType, int device);
	protected native void finalize();
}