package com.glasssix.Irisvian;

public class Search {
	static {
		System.loadLibrary("Irisvian-java")
	}
	
	private long mObject;
	
	public Search(float[][] baseData) {
		init(baseData);
	}
	
	public Search(int dimension) {
		init(dimension);
	}
	
	public native void loadGraph(String graphPath);
	public native void loadGraph(String graphPath, String basedataPath);
	public native void optimizeGraph();
	public native void searchVector(float[][] queryData, int topK, int[][] returnIDs, float[][] returnSimilarities);
	public native void saveResult(String resultPath, int[][] returnIDs);
	private native void init(float[][] baseData);
	private native void init(int dimension);
	protected native void finalize();
}