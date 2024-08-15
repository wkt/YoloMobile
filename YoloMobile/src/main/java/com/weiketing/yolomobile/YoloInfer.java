package com.weiketing.yolomobile;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.os.Environment;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class YoloInfer {
    private static final String TAG = "YoloInferJava";

    static public final int BIND_CPU_ALL = 0;
    static public final int BIND_CPU_LITTLE = 1;
    static public final int BIND_CPU_BIG = 2;


    static {
       System.loadLibrary("yolomobile");
    }

    static class Point extends android.graphics.Point{
        final float p;

        protected Point(int x,int y,float p) {
            super(x,y);
            this.p = p;
        }

        @Override
        public String toString() {
            return "Point{" +
                    "p=" + p +
                    ", x=" + x +
                    ", y=" + y +
                    '}';
        }
    }

    static public class Box {
        final public Rect     rect = new Rect(-1,-1,-1,-1);
        final public int      c;    // class id
        final public float    p;    // prob
        final public String   n;    // class name
        final public List<Point> keyPoints = new ArrayList<>();

        protected Box(int x,int y, int w,int h,int c, String n,float p) {
            this.rect.left = x;
            this.rect.top = y;
            this.rect.bottom = y+h;
            this.rect.right = x+w;
            this.c = c;
            this.n = n;
            this.p = p;
        }

        @Override
        public String toString() {
            return "Box{" +
                    " x=" + rect.left +
                    ", y=" + rect.top +
                    ", w=" + rect.width() +
                    ", h=" + rect.height() +
                    ", c=" + c +
                    ", p=" + p +
                    ", keyPoints=" + keyPoints +
                    '}';
        }
    }

    private long _obj = 0;
    private int inputSize = 416;
    private String paramName; // param file in assets
    private String modelName; // model file in assets
    private String inputName = "images";
    private final Context ctx;
    private String name;

    private native void init();
    private native void load_model(AssetManager asset,String param,String bin);
    private native void load_model_fullpath(String paramPath,String binPath);
    private native float[] forward(Bitmap img);
    private native void update(String key,String value);
    private native void bind_cpu(int flag);
    private native void release();

    static private String FILES_DIR = null;
    static private String EXTERNAL_FILES_DIR= null;

    static private String EXTERNAL_DIR = null;

    private String names[] = null; //class names
    public YoloInfer(Context ctx,String name){
        this.ctx = ctx.getApplicationContext();
        _obj = 0;
        this.name = name;
        if (FILES_DIR == null) {
            EXTERNAL_FILES_DIR = ctx.getExternalFilesDir(null).getAbsolutePath();
            FILES_DIR = ctx.getFilesDir().getAbsolutePath();
            EXTERNAL_DIR = Environment.getExternalStorageDirectory().getAbsolutePath();
        }
        //Log.e(TAG,"EXTERNAL_FILES_DIR: "+EXTERNAL_FILES_DIR);
        //Log.e(TAG,"FILES_DIR: "+FILES_DIR);
        //Log.e(TAG,"EXTERNAL_DIR: "+EXTERNAL_DIR);
    }

    public YoloInfer(Context ctx){
        this(ctx,"Yolo");
    }

    /**
     *
     * @param paramName .param file's name in assets or a full path in local file system
     */
    public void setParamName(String paramName) {
        /**
         * Must call before loadModel
         */
        this.paramName = paramName;
    }

    /**
     *
     * @param modelName model .bin file's name in assets or a full path in local file system
     */
    public void setBinName(String modelName) {
        /**
         * Must call before loadModel
         */
        this.modelName = modelName;
    }

    public void loadModel(){
        init();
        _update("input_name",inputName);
        _update("input_size",inputSize);
        if(paramName.startsWith(File.separator)){
            load_model_fullpath(paramName,modelName);
        }else {
            load_model(ctx.getAssets(), paramName, modelName);
        }
    }

    public void loadFromJson(String json,String jsonDir){
        try {
            json = json.replaceAll("@FILES_DIR@",FILES_DIR)
                    .replaceAll("@EXTERNAL_DIR@",EXTERNAL_DIR)
                    .replaceAll("@SDCARD_DIR@",EXTERNAL_DIR)
                    .replaceAll("@SDCARD@",EXTERNAL_DIR)
                    .replaceAll("@EXTERNAL_FILES_DIR@",EXTERNAL_FILES_DIR);
            JSONObject obj = new JSONObject(json);
            String p = obj.optString("param","yolo.param");
            if(jsonDir!=null){
                paramName = new File(jsonDir,p).getAbsolutePath();
            }else{
                paramName = p;
            }
            p = obj.optString("bin","yolo.bin");
            if(jsonDir!=null){
                modelName = new File(jsonDir,p).getAbsolutePath();
            }else{
                modelName = p;
            }
            if(obj.has("name")){
                name = obj.optString("name",this.name);
            }
            inputName = obj.optString("input_name",inputName);
            inputSize = obj.optInt("input_size",inputSize);
            loadModel();

            if(obj.has("names")){
                JSONArray array = obj.optJSONArray("names");
                int n = array.length();
                if(n>0){
                    names = new String[n];
                    for(int i=0;i<n;i++){
                        names[i] = array.optString(i);
                    }
                }
            }else{
                names = null;
            }
            List<String> ignores = Arrays.asList("bin", "param", "input", "input_size","name","names");
            Iterator<String> it = obj.keys();
            while (it.hasNext() ) {
                String k = it.next();
                if(ignores.contains(k))continue;
                _update(k,obj.opt(k));
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

    public void loadFromConfig(String cfgFile){
        File file = new File(cfgFile);
        File dir = file.getParentFile();
        loadFromJson(Util.readText(file),dir.getAbsolutePath());
    }

    public void loadFromConfigAssets(String name){
        try {
            loadFromJson(Util.readText(ctx.getAssets().open(name)),null);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public List<Box>detect(Bitmap bitmap){
        float[] nums = forward(bitmap);
        List<Box>boxes = new ArrayList<>();
        if(nums == null)return boxes;
        int n = (int) nums[0];
        int d = (int) nums[1];
        int c = 0;
        for(int i=0;i<n;i++){
            int i0 = 2+i * d;
            c = (int)nums[i0+4];
            Box b = new Box((int)nums[i0],(int)nums[i0+1],(int)nums[i0+2],(int)nums[i0+3],c,nameByIndex(c),nums[i0+5]);
            int nkpt = (d - 6) / 3;
            if(nkpt>0){
                for(int j=0;j<nkpt;j++){
                    int j0=i0+6+j*3;
                    Point p = new Point((int)nums[j0],(int)nums[j0+1],nums[j0+2]);
                    b.keyPoints.add(p);
                }
            }
            boxes.add(b);
        }
        //Log.e(TAG,"boxes:"+boxes);
        return boxes;
    }

    private<T> void _update(String k,T v){
        //Log.d(TAG,"k: "+k+", v: "+v);
        if(k.equalsIgnoreCase("outputs")){
            JSONArray array = (JSONArray) v;
            for(int i=0;i<array.length();i++){
                JSONObject o = array.optJSONObject(i);
                if(!o.has("name"))continue;
                try {
                    addOutput(o.optString("name"),o.optInt("stride"),o.optJSONArray("anchors").join(","));
                } catch (JSONException e) {
                    e.printStackTrace();
                }
            }
            return;
        }
        Object o = v;
        if(v instanceof JSONArray){
            try {
                o = ((JSONArray) v).join(",");
            } catch (JSONException e) {
                e.printStackTrace();
            }
        }
        update(k,String.valueOf(o));
    }

    public void setBoxThreshold(float thr){
        update("box_thr",String.valueOf(thr));
    }

    public void setIouThreshold(float thr){
        update("iou_thr",String.valueOf(thr));
    }

    public String getName() {
        return name;
    }

    public int getInputSize() {
        return inputSize;
    }

    public void setInputSize(int inputSize) {
        this.inputSize = inputSize;
        if (_obj == 0)return;
        _update("input_size",inputSize);
    }

    public void setInputName(String inputName) {
        this.inputName = inputName;
        if (_obj == 0)return;
        _update("input_name",inputName);
    }

    public void addOutput(String name,int stride,String anchors){
        _update("output_name",name);
        _update("output_stride",stride);
        _update("output_anchors",anchors);
    }

    public void addOutput(String name,int stride,Float[] anchors){
        addOutput(name,stride,Util.join(",",anchors));
    }

    public void setNumKeypoint(int num){
        _update("nkpt",num);
    }

    public void bindCPU(int flag){
        bind_cpu(flag);
    }

    public String nameByIndex(int i){
        if(names == null || names.length == 0 || i>=names.length)return String.format("%d",i);
        return names[i];
    }

    @Override
    protected void finalize() throws Throwable {
        release();
        super.finalize();
    }

    static public final int[]  COCO17_SKELETON = new int[]{
            0, 1,
            0, 2,
            1, 3,
            2, 4,
            0, 5,
            0, 6,
            5, 6,
            5, 7,
            6, 8,
            7, 9,
            8, 10,
            5, 11,
            6, 12,
            11, 12,
            11, 13,
            12, 14,
            13, 15,
            14, 16,
    };
    static public final int[] FACE5LAND_MARK = new int[]{
            0,1,
            0,2,
            0,3,
            0,4,
            1,2,
            2,4,
            4,3,
            3,1,
    };

    static public void draw(Canvas canvas, List<Box>boxes,boolean kptIndex,boolean boxInfo, Paint paint){
        if(boxes==null)return;
        if(paint == null){
            paint = new Paint();
        }
        int kptI= 0;

        for(Box b:boxes){
            paint.setStrokeWidth(1);
            paint.setTextSize(16);
            paint.setStyle(Paint.Style.STROKE);
            paint.setColor(Color.GREEN);
            canvas.drawRect(b.rect,paint);
            paint.setStyle(Paint.Style.FILL);

            if(boxInfo) {
                String txt = String.format("c: %s, %.3f", b.n, b.p);
                Rect tr = new Rect();
                paint.getTextBounds(txt, 0, txt.length(), tr);
                canvas.drawRect(b.rect.left, b.rect.top-4, b.rect.left + tr.width() + 8, b.rect.top + tr.height()+4, paint);
                paint.setColor(Color.BLACK);
                canvas.drawText(txt, b.rect.left + 4, b.rect.top + tr.height() - 4, paint);
            }
            kptI = 0;
            int nkpt = b.keyPoints.size();
            if(nkpt>0){
                int[] pairs = new int[0];
                if(nkpt == 5){
                    pairs = FACE5LAND_MARK;
                } else if (nkpt == 17) {
                    pairs = COCO17_SKELETON;
                }
                paint.setColor(Color.BLUE);
                for(int i=0;i<pairs.length/2;i++){
                    int i0 = pairs[i * 2];
                    int i1 = pairs[i * 2 + 1];
                    Point p0 = b.keyPoints.get(i0);
                    Point p1 = b.keyPoints.get(i1);
                    if(p0.p>0 && p1.p>0){
                        canvas.drawLine(p0.x,p0.y,p1.x,p1.y,paint);
                    }
                }
            }
            for(Point p:b.keyPoints){
                paint.setColor(Color.WHITE);
                canvas.drawCircle(p.x,p.y,2,paint);
                paint.setColor(Color.RED);
                canvas.drawCircle(p.x,p.y,1,paint);
                if(kptIndex) {
                    canvas.drawText("" + kptI, p.x, p.y, paint);
                    kptI+=1;
                }
            }
        }
    }

    static public void draw(Canvas canvas, List<Box>boxes, Paint paint){
        draw(canvas,boxes,true,true,paint);
    }
}
