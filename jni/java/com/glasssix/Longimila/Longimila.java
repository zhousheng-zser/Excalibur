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

	public byte[] alignFacebyMat(long grayNativeObj, FaceRectwithFaceInfo[] face_info)
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
		return alignFacebyMat(grayNativeObj, bbox, landmark);
	}

	public byte[] alignFacebyMetaData(byte[] metadata, int width, int height, FaceRectwithFaceInfo[] face_info)
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
		return alignFacebyMetaData(metadata, width, height, bbox, landmark);
	}


//	public static native void extract_biggest_faceinfo(FaceRectwithFaceInfo[] face_info,
//			int[][] bboxes, int[][] landmarks);
//	public static native void extract_faceinfo(FaceRectwithFaceInfo[] face_info,
//			int[][] bboxes, int[][] landmarks);
	public static native String getVersion();
	public native FaceRect[] detectbyMat(long grayNativeObj, int minSize, float scale, int minNeighbors);
	public native FaceRectwithFaceInfo[] detectwithInfobyMat(long grayNativeObj, int minSize, float scale, int minNeighbors, int order);
	public native FaceRectwithFaceInfo[] detectExbyMat(long matNativeObj, int minSize, float[] threshold, float factor, int stage, int order);
	public native FaceRectwithFaceInfo[] detectExbyMatMobile(long matNativeObj, int minSize, float[] threshold, float factor, int stage, int order);
	public native FaceRect[] detectbyMetaData(byte[] metadata, int width, int height, int minSize, float scale, int minNeighbors);
	public native FaceRectwithFaceInfo[] detectwithInfobyMetaData(byte[] metadata, int width, int height, int minSize, float scale, int minNeighbors, int order);
	public native FaceRectwithFaceInfo[] detectExbyMetaData(byte[] metadata, int width, int height, int minSize, float[] threshold, float factor, int stage, int order);
	public native FaceRectwithFaceInfo[] detectExbyMetaDataMobile(byte[] metadata, int width, int height, int minSize, float[] threshold, float factor, int stage, int order);
	public native Match_Rectval[] match(FaceRect[] faceRect, int frame_extract_frequency, float distance_fractor);
	private native byte[] alignFacebyMat(long grayNativeObj, int[][] bbox, int[][] landmarks);
	private native byte[] alignFacebyMetaData(byte[] metadata, int width, int height, int[][] bbox, int[][] landmarks);
	public native byte[] alignSingleFacebyMat(long grayNativeObj);
	public native byte[] alignSingleFacebyMetaData(byte[] metadata, int width, int height);
	private native void init(int device);
	private native void set(int detectionType, int device);
	protected native void finalize();
}