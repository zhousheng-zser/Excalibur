package com.glasssix.Irisvika;

public class Irisvika {
	static {
		System.loadLibrary("Irisvika");
	}
	
	private long mObject;
	
	public Irisvika(int max_items, String new_save_path, String tmp_path) {
		init(max_items, new_save_path, tmp_path);
	}
	
	private native void init(int max_items, String new_save_path, String tmp_path);
	protected native void finalize();
	public native String save_path();
	public native String tmp_path();
	public native void build(String[] files);
	public native knn_search_result[] search(float[] feature, int top);
	public native String[] delete_features(String[] keys);
	public native String[] delete_feature(String key);
	public native void add_features(knn_mapping_data[] data);
	public native void add_feature(knn_mapping_data data);
	public native void update(knn_mapping_data data);
	public native void update_more(knn_mapping_data[] data);
}