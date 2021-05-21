package com.example.myapplication;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;

import java.io.FileNotFoundException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {
    private static int RESULT_LOAD_IMAGE = 1;
    Bitmap srcBitmap = null;
    Bitmap dstBitmap = null;
    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    public void btnAGCIE_click(View view){
        AGCIE(srcBitmap,dstBitmap);
        View nImg = findViewById(R.id.imageViewOutput);
        ((ImageView)nImg).setImageBitmap(dstBitmap);
    }

    public void btnBIMEF_click(View view){
        BIMEF(srcBitmap,dstBitmap);
        View nImg = findViewById(R.id.imageViewOutput);
        ((ImageView)nImg).setImageBitmap(dstBitmap);
    }

    public void btnAGCWD_click(View view){

        EditText mEdit;
        mEdit   = (EditText)findViewById(R.id.editTextNumber);
        double alpha = Double.parseDouble(mEdit.getText().toString());
        AGCWDInput(srcBitmap,dstBitmap,alpha);
        View nImg = findViewById(R.id.imageViewOutput);
        ((ImageView)nImg).setImageBitmap(dstBitmap);
    }

    public void btnIAGCWD_click(View view){

        EditText mEdit,mEdit1,mEdit2,mEdit3,mEdit4;
        mEdit   = (EditText)findViewById(R.id.editTextNumber);
        mEdit1   = (EditText)findViewById(R.id.editTextNumber1);
        mEdit2   = (EditText)findViewById(R.id.editTextNumber2);
        mEdit3   = (EditText)findViewById(R.id.editTextNumber3);
        mEdit4   = (EditText)findViewById(R.id.editTextNumber4);

        double alpha_dimmed = Double.parseDouble(mEdit.getText().toString());
        double alpha_bright = Double.parseDouble(mEdit1.getText().toString());
        int T_t = Integer.parseInt(mEdit2.getText().toString());
        double tau_t = Double.parseDouble(mEdit3.getText().toString());
        double tau = Double.parseDouble(mEdit4.getText().toString());


        IAGCWD(srcBitmap,dstBitmap,alpha_dimmed,alpha_bright,T_t,tau_t,tau);
        View nImg = findViewById(R.id.imageViewOutput);
        ((ImageView)nImg).setImageBitmap(dstBitmap);
    }

    public void btnAGCIEDSUS_click(View view){
        AGCIEDSUS(srcBitmap,dstBitmap);
        View nImg = findViewById(R.id.imageViewOutput);
        ((ImageView)nImg).setImageBitmap(dstBitmap);
    }

    public void btnBIMEFDSUS_click(View view){

        EditText mEdit,mEdit1,mEdit2;
        mEdit   = (EditText)findViewById(R.id.editTextNumber);
        mEdit1   = (EditText)findViewById(R.id.editTextNumber1);
        mEdit2   = (EditText)findViewById(R.id.editTextNumber2);
        double mu = Double.parseDouble(mEdit.getText().toString());
        double a = Double.parseDouble(mEdit1.getText().toString());
        double b = Double.parseDouble(mEdit2.getText().toString());

        BIMEFDSUSInput(srcBitmap,dstBitmap,mu,a,b);
        View nImg = findViewById(R.id.imageViewOutput);
        ((ImageView)nImg).setImageBitmap(dstBitmap);
    }

    public void btnAGCWDDSUS_click(View view){
        AGCWDDSUS(srcBitmap,dstBitmap);
        View nImg = findViewById(R.id.imageViewOutput);
        ((ImageView)nImg).setImageBitmap(dstBitmap);
    }

    public void btnLIME_click(View view){
        LIME(srcBitmap,dstBitmap);
        View nImg = findViewById(R.id.imageViewOutput);
        ((ImageView)nImg).setImageBitmap(dstBitmap);
    }



    public void btnLoad_click(View view){
        openGallery();
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        startActivityForResult(Intent.createChooser(intent,"Pick an image"),RESULT_LOAD_IMAGE);
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data){
        if (resultCode == RESULT_OK && requestCode == RESULT_LOAD_IMAGE){
            ImageView imageView = findViewById(R.id.imgView);
            try{
                InputStream inputStream = getContentResolver().openInputStream(data.getData());
                srcBitmap = BitmapFactory.decodeStream(inputStream);
                dstBitmap = srcBitmap.copy(srcBitmap.getConfig(), true);
                imageView.setImageBitmap(srcBitmap);
            }
            catch (FileNotFoundException e)
            {
                e.printStackTrace();
            }
        }
    }

    public native String stringFromJNI();
    public native void myFlip(Bitmap bitmapIn,Bitmap bitmapOut);
    public native void AGCIE(Bitmap bitmapIn,Bitmap bitmapOut);
    public native void LIME(Bitmap bitmapIn,Bitmap bitmapOut);
    public native void BIMEF(Bitmap bitmapIn,Bitmap bitmapOut);
    public native void AGCWD(Bitmap bitmapIn,Bitmap bitmapOut);
    public native void AGCWDInput(Bitmap bitmapIn,Bitmap bitmapOut, double alpha);
    public native void AGCIEDSUS(Bitmap bitmapIn,Bitmap bitmapOut);
    public native void AGCWDDSUS(Bitmap bitmapIn,Bitmap bitmapOut);
    public native void BIMEFDSUS(Bitmap bitmapIn,Bitmap bitmapOut);
    public native void BIMEFDSUSInput(Bitmap bitmapIn,Bitmap bitmapOut,double mu,double a,double b);
    public native void IAGCWD(Bitmap bitmapIn,Bitmap bitmapOut,double alpha_dimmed , double alpha_bright , int T_t , double tau_t, double tau ) ;

    //public native fun myBlur(Bitmap bitmapIn,Bitmap bitmapOut, Float sigma);
    //public native fun myFlip(Bitmap bitmapIn,Bitmap bitmapOut);
}