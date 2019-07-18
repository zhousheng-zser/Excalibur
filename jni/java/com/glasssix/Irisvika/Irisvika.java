package com.glasssix.Irisvika;

public class Irisvika {
	static {
		System.loadLibrary("Irisvika");
	}
	
	private long mObject;
	
	public Irisvika(float[][] baseData) {
		init_withData(baseData);
	}
	
	public Irisvika(int dimension) {
		init(dimension);
	}
	
	public native void loadGraph(String graphPath);
	public native void loadGraph(String graphPath, String basedataPath);
	public native void optimizeGraph();
	public native void searchVector(float[][] queryData, int topK, int[][] returnIDs, float[][] returnSimilarities);
	public native void saveResult(String resultPath, int[][] returnIDs);
	private native void init_withData(float[][] baseData);
	private native void init(int dimension);
	protected native void finalize();
}