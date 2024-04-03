package com.weiketing.yolomobile.example;

import android.app.ActionBar;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.os.Bundle;
import android.os.SystemClock;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.PopupMenu;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.davemorrissey.labs.subscaleview.ImageSource;
import com.davemorrissey.labs.subscaleview.SubsamplingScaleImageView;
import com.weiketing.yolomobile.example.R;
import com.weiketing.yolomobile.YoloInfer;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private YoloInfer infer;
    private SubsamplingScaleImageView imageView;
    private Button btnNext;
    private Button btnFPS;
    private TextView tvInfo;
    private TextView tvTitle;
    private List<String> images = new ArrayList<>();
    private int idx = 0;
    private boolean testFPS = false;
    private TextView modeName;
    

    static private String[] models = new String[]{
            "yolov7-tiny",
            "yolov8n-face-384",
            "yolov8n1-face-448",
            "yolov5s6_640_ti_lite_54p9_82p2_opt",
            "yolov5n",
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = findViewById(R.id.imageView);
        btnFPS = findViewById(R.id.btnFPS);
        btnNext = findViewById(R.id.btnNext);
        tvInfo = findViewById(R.id.tvInfo);
        tvTitle = findViewById(R.id.tvTitle);
        modeName = findViewById(R.id.modeName);

        btnNext.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                testFPS = false;
                doNext();
                btnFPS.setEnabled(true);
                tvInfo.setText("");
            }
        });
        btnFPS.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                btnFPS.setEnabled(false);
                doTestFPS();
            }
        });
        imageView.setMaxScale(3.0f);
        listModels();
        listImages();
        //images.add("imgs/640.png");
        loadModel(models[0]);
        doNext();
        modeName.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                showPopupMenu(view);
            }
        });
    }

    private void doTestFPS() {
        testFPS = true;
        new Thread(new Runnable() {
            @Override
            public void run() {
                int j = 0;
                long t = 0;
                long n = 0;
                long t0 = 0;
                long nt = 0;
                infer.bindCPU(YoloInfer.BIND_CPU_BIG);
                while (testFPS){
                    if(j>=images.size())j=0;
                    InputStream ins = null;
                    try {
                        ins = getAssets().open(images.get(j));
                        Bitmap bm = BitmapFactory.decodeStream(ins);
                        ins.close();
                        t0 = SystemClock.elapsedRealtime();
                        List<YoloInfer.Box> boxes = infer.detect(bm);
                        nt = SystemClock.elapsedRealtime();
                        t += nt-t0;
                        n += 1;
                        Bitmap img = Bitmap.createBitmap(bm.getWidth(),bm.getHeight(), Bitmap.Config.ARGB_8888);
                        Paint paint = new Paint();
                        Canvas canvas = new Canvas(img);
                        canvas.drawBitmap(bm,0,0,paint);
                        YoloInfer.draw(canvas,boxes,paint);
                        double fps = n / (t / 1000.);
                        int finalJ = j;
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                if(!testFPS)return;
                                setTitle(images.get(finalJ));
                                imageView.setImage(ImageSource.bitmap(img));
                                String txt = String.format("Model: %s\nInput size: %d\nInfer FPS: %.3f",infer.getName(),infer.getInputSize(),fps);
                                tvInfo.setText(txt);
                            }
                        });
                        j+=1;
                        if(n>128){
                            n=0;
                            t=0;
                        }
                        Thread.sleep(100);
                    } catch (IOException e) {
                        e.printStackTrace();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }

                }

            }
        }).start();
    }

    private void listImages(){
        try {
            images.clear();
            String path = "imgs";
            String[] ll = getAssets().list(path);
            for(String n:ll){
                String ln = n.toLowerCase();
                if(ln.endsWith(".jpeg") || ln.endsWith(".jpg") || ln.endsWith(".bmp") || ln.endsWith(".png") || ln.endsWith(".webm")){
                    images.add(path+"/"+n);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void listModels(){
        List<String>names = new ArrayList<>();
        try {
            String path = "";
            String[] ll = getAssets().list(path);
            for(String n:ll){
                String ln = n.toLowerCase();
                if(ln.endsWith(".json")){
                    names.add(n);
                }
            }
            models = names.toArray(new String[0]);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void setTitle(CharSequence title) {
        super.setTitle(title);
        ActionBar abr = getActionBar();
        if(abr!=null){
            abr.setTitle(title);
        }
        androidx.appcompat.app.ActionBar sabr = getSupportActionBar();
        if(sabr!=null) {
            sabr.setTitle(title);
        }
        tvTitle.setText(title);
    }

    protected void doNext(){
        try {
            if(idx>=images.size()){
                idx = 0;
            }
            btnNext.setEnabled(false);
            InputStream ins = getAssets().open(images.get(idx));
            Bitmap bm = BitmapFactory.decodeStream(ins);
            ins.close();
            imageView.setImage(ImageSource.bitmap(bm));
            new Thread(new Runnable() {
                @Override
                public void run() {
                    List<YoloInfer.Box> boxes = infer.detect(bm);
                    Bitmap img = Bitmap.createBitmap(bm.getWidth(),bm.getHeight(), Bitmap.Config.ARGB_8888);
                    Paint paint = new Paint();
                    Canvas canvas = new Canvas(img);
                    canvas.drawBitmap(bm,0,0,paint);
                    YoloInfer.draw(canvas,boxes,paint);
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            setTitle(images.get(idx));
                            imageView.setImage(ImageSource.bitmap(img));
                            idx += 1;
                            btnNext.setEnabled(true);
                        }
                    });
                }
            }).start();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void showPopupMenu(View anchor){
        PopupMenu popup = new PopupMenu(this,anchor);
        Menu menu = popup.getMenu();
        for(int i=0;i<models.length;i++){
            menu.add(0,i,i,models[i]);
        }
        popup.setOnMenuItemClickListener(new PopupMenu.OnMenuItemClickListener() {
            @Override
            public boolean onMenuItemClick(MenuItem menuItem) {
                loadModel(models[menuItem.getItemId()]);
                btnNext.performClick();
                return false;
            }
        });
        popup.show();
    }

    private void loadModel(String name)
    {

        infer = new YoloInfer(this);
        /*
        infer.setInputSize(384);
        infer.setParamName("last.param");
        infer.setBinName("last.bin");
        infer.loadModel();
        infer.setBoxThreshold(0.45f);
        infer.setIOUThreshold(0.35f);
        infer.setNumKeypoint(5);
        infer.addOutput("397",8,new Float[]{10f,13f, 16f,30f, 33f,23f});
        infer.addOutput("458",16,new Float[]{30f,61f, 62f,45f, 59f,119f});
        infer.addOutput("519",32,new Float[]{116f,90f, 156f,198f, 373f,326f});

         */
        //infer.loadFromConfigAssets("cfg.json");
        modeName.setText("Model cfg: "+name);
        if (!name.toLowerCase().endsWith(".json")){
            name += ".json";
        }
        infer.loadFromConfigAssets(name);
    }

}