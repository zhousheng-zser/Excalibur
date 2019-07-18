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

	public byte[] alignFace(long grayNativeObj, FaceRectwithFaceInfo[] face_info)
	{
		int[][] bbox = new int[face_info.length][];
		int[][] landmark = new int[face_info.length][];
		for (int i = 0; i < face_info.length; i++)
		{
			bbox[i] = new int[4];
			bbox[i][0] = face_info[i].x;
			bbox[i][1] = face_info[i].y;
			bbox[i][2] = face_info[i].width;
			bbox[i][3] = face_info[i].height;
			landmark[i] = new int[10];
			for (int j = 0; j < 5; j++)
			{
				landmark[i][j * 2 + 0] = face_info[i].pts[j].x;
				landmark[i][j * 2 + 1] = face_info[i].pts[j].y;
			}
		}
		return alignFace(grayNativeObj, bbox, landmark);
	}


	
	public static native String getVersion();
	public native FaceRect[] detect(long grayNativeObj, int minSize, float scale, int minNeighbors);
	public native FaceRectwithFaceInfo[] detectwithInfo(long grayNativeObj, int minSize, float scale, int minNeighbors, int order);
	public native Match_Rectval[] match(FaceRect[] faceRect, int frame_extract_frequency);
	public native byte[] alignFace(long grayNativeObj, int[][] bbox, int[][] landmarks);
	public native byte[] alignSingleFace(long grayNativeObj);
	private native void init(int device);
	private native void set(int detectionType, int device);
	protected native void finalize();
}