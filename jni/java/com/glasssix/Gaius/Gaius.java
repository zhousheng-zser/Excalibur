package com.glasssix.Gaius;

public class GaiusFeature {
	static {
		System.loadLibrary("Gaius-java")
	}
	
	private long mObject;
	
	public GaiusFeature(int device) {
		init(device);
	}
	
	public static native String getVersion();
	public native float[] Forward(long MatNativeObj, int order);
	private native void init(int device);
	protected native void finalize();
}