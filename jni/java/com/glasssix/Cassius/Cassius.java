package com.glasssix.Cassius;

public class CassiusFeature {
	static {
		System.loadLibrary("Cassius-java")
	}
	
	private long mObject;
	
	public CassiusFeature(int device) {
		init(device);
	}
	
	public static native String getVersion();
	public native float[] Forward(long MatNativeObj, int order);
	private native void init(int device);
	protected native void finalize();
}