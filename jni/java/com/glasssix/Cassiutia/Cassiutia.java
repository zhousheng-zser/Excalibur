package com.glasssix.Cassiutia;

public class Cassiutia {
	static {
		System.loadLibrary("Cassiutia");
	}
	
	private long mObject;
	
	public Cassiutia(int device) {
		init(device);
	}
	
	public static native String getVersion();
	public native float[] Forward(long MatNativeObj, int order);
	private native void init(int device);
	protected native void finalize();
}