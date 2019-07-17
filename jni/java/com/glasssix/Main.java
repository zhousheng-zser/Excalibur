package com.glasssix;
import com.glasssix.Gaiulinya.Gaiulinya;
import com.glasssix.Cassiutia.Cassiutia;
import com.glasssix.Irisvika.Irisvika;
import com.glasssix.Longimila.Longimila;

public class Main {

    public static void main(String[] args) {
	// write your code here
        System.out.println("Hello World!");
        System.out.println(Gaiulinya.getVersion());
        String v = Cassiutia.getVersion();
        System.out.println(v);
        Irisvika s = new Irisvika(512);
        Longimila ll = new Longimila(-1);
        //s.loadGraph("D:\\Research\\Excalibur\\data\\map512.bin");
        System.out.println("Load success!");
    }
}
