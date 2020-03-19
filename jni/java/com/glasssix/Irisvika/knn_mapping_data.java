package com.glasssix.Irisvika;

public class knn_mapping_data {
	
	public float[] feature;
	public byte[] key;
	
	public knn_mapping_data() {
	}
	
	public knn_mapping_data(knn_mapping_data other) {
		this.feature = other.feature;
		this.key = other.key;
	}
}