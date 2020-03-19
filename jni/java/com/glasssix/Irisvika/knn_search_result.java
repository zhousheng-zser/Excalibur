package com.glasssix.Irisvika;

public class knn_search_result {
	public knn_mapping_data data;
	public float distance_in_percentage;
	
	public knn_search_result() {
	}
	
	public knn_search_result(knn_search_result other) {
		this.data = other.data;
		this.distance_in_percentage = other.distance_in_percentage;
	}
	
	public knn_search_result(knn_mapping_data data, float distance_in_percentage) {
		this.data = data;
		this.distance_in_percentage = distance_in_percentage;
	}
}