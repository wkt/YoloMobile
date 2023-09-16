package com.weiketing.yolomobile;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;

public class Util {


    private static final int BUFFER_SIZE = 4096;

    static public<T> String join(String sep, T[] array){
        if(array == null)return "";
        StringBuilder sb = new StringBuilder();
        for(T t:array){
            if(sb.length() > 0){
                sb.append(sep);
            }
            sb.append(String.valueOf(t));
        }
        return sb.toString();
    }

    static public byte[] readStream(InputStream ins,boolean autoClose){
        ByteArrayOutputStream bao = new ByteArrayOutputStream();
        byte[] buf = new byte[BUFFER_SIZE];
        int n=0;
        while (true){
            try {
                n = ins.read(buf);
                if(n <= 0)break;
                bao.write(buf,0,n);
            } catch (IOException e) {
                e.printStackTrace();
                break;
            }
        }
        if(autoClose){
            try {
                ins.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        byte[] ret = bao.toByteArray();
        try {
            bao.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ret;
    }

    static public byte[] readStream(InputStream ins)
    {
        return readStream(ins,true);
    }

    static public String readText(File file){
        try {
            return readText(new FileInputStream(file));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return "";
    }

    static public String readText(InputStream ins){
        return new String(readStream(ins), Charset.defaultCharset());
    }
}
