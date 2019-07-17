package com.glasssix.Longimila;

public class FaceRectwithFaceInfo extends FaceRect {
	public Point[] pts;
	public float yaw;
	public float pitch;
	public float roll;
	
	public FaceRectwithFaceInfo() {
		pts = new Point[5];
		yaw = 0.0f;
		pitch = 0.0f;
		roll = 0.0f;
	}
	
	public FaceRectwithFaceInfo(FaceRect rect) {
		super.x = rect.x;
		super.y = rect.y;
		super.width = rect.width;
		super.height = rect.height;
		super.neighbors = rect.neighbors;
		super.confidence = rect.confidence;
		pts = new Point[5];
		yaw = 0.0f;
		pitch = 0.0f;
		roll = 0.0f;
	}
}