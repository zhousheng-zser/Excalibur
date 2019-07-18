package com.glasssix.Gaiulinya;

public class Gaiulinya {
	static {
		System.loadLibrary("Gaiulinya");
	}
	
	private long mObject;
	
	public Gaiulinya(int device) {
		init(device);
	}
	
	public static native String getVersion();
	public native float[] Forward(long MatNativeObj, int order);
	private native void init(int device);
	protected native void finalize();
}