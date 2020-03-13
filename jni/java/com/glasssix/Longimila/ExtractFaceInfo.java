package com.glasssix.Longimila;

public class ExtractFaceInfo {
	public int[][] bbox;
	public int[][] landmark;
	
	
	public ExtractFaceInfo() {
	}
	
	public ExtractFaceInfo(int[][] bbox, int[][] landmark) {
		this.bbox = bbox;
		this.landmark = landmark;
	}
}