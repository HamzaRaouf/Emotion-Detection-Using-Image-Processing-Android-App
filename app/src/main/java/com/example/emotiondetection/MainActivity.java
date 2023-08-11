package com.example.emotiondetection;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.emotiondetection.ml.Model;

import org.jetbrains.annotations.Nullable;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {
    TextView result, confidence;
    ImageView imageView;
    TextView picture;
    int imageSize = 48;   // which is given by our Trained Model
    private static final int REQUEST_CAMERA_PERMISSION = 1;
    private static final int REQUEST_IMAGE_CAPTURE = 2;
    private static final int REQUEST_IMAGE_PICK = 3;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);

        picture.post(() -> picture.performClick());

        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    //Request camera permission if we don't have it.
                    requestPermissions(new String[]{android.Manifest.permission.CAMERA}, 100);
                }
            }
        });

    }


    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == 1 && resultCode == RESULT_OK) {

            // receive bitmap from data intent
            Bitmap image = (Bitmap) data.getExtras().get("data");
            // crop image

            int dimentions = Math.min(image.getWidth(),image.getHeight());
            imageView.setImageBitmap(image);

            // create Image as a size of our model Requirement
            image = Bitmap.createScaledBitmap(image,imageSize, imageSize,false);
            // Resize the image to the desired input size

// Convert the resized image to grayscale
            Bitmap grayscaleImage = convertToGrayscale(image);

            classifyImage(grayscaleImage);



        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private Bitmap convertToGrayscale(Bitmap originalBitmap) {
        Bitmap grayscaleBitmap = Bitmap.createBitmap(originalBitmap.getWidth(), originalBitmap.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(grayscaleBitmap);
        Paint paint = new Paint();
        ColorMatrix matrix = new ColorMatrix();
        matrix.setSaturation(0); // Convert to grayscale
        ColorMatrixColorFilter filter = new ColorMatrixColorFilter(matrix);
        paint.setColorFilter(filter);
        canvas.drawBitmap(originalBitmap, 0, 0, paint);
        return grayscaleBitmap;
    }

    private void classifyImage(Bitmap image) {
        try {
            Model model = Model.newInstance(getApplicationContext());
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 48, 48, 1}, DataType.FLOAT32);


// Create the input buffer for the model
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            for (int pixelValue : intValues) {
                // Extract the grayscale pixel value and normalize it
                float normalizedValue = (pixelValue & 0xFF) / 255.0f;
                byteBuffer.putFloat(normalizedValue);
            }

//            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4*imageSize*imageSize*1);
//            byteBuffer.order(ByteOrder.nativeOrder());
//            int [] intValues = new int[imageSize * imageSize]; // get pixl size of Image
//            image.getPixels(intValues,0,image.getWidth(),0,0,image.getWidth(),image.getHeight());
//             int pixel=0;
//
//
//            for (int i = 0; i < imageSize; ++i) {
//                for (int j = 0; j < imageSize; ++j) {
//                    final int val = intValues[pixel++];
//                    byteBuffer.putFloat((val >> 16 & 0xFF) / 255.0f);
//                    byteBuffer.putFloat((val >> 8 & 0xFF) / 255.0f);
//                    byteBuffer.putFloat((val & 0xFF) / 255.0f);
//                }
//            }

            // check which pixl is On
//             for(int i=0; i<imageSize ; i++){
//                 for (int j=0; j<imageSize; j++){
//
//                     int val = intValues[pixel++]; // RGB
//                     byteBuffer.putFloat((( val >> 16 ) & 0xFF)*(1.f/255.f));
//                     byteBuffer.putFloat((( val >> 8 ) & 0xFF)*(1.f/255.f));
//                     byteBuffer.putFloat(( val & 0xFF ) *(1.f/255.f));
//
//                 }
//             }


            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            System.out.println("====>> Models Ouput :"+outputFeature0);
            System.out.println("====>> Models Ouput size  :"+outputFeature0.getFlatSize());

            // Manipulate OutPut Taked from Model

            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            String[] classes ={"Angry  \uD83D\uDE21", "Disgust \uD83D\uDE2D", "Scared \uD83D\uDE31", "Happy \uD83D\uDE00", "Sad \uD83D\uDE41", "Surprised \uD83D\uDE32", "Neutral \uD83D\uDE10"};
            for (int i=0; i<confidences.length; i++){
                System.out.println("====>>"+classes[i]+":"+confidences[i]);
                if(confidences[i] > maxConfidence){
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            System.out.println("====>> Maximum Value Index :"+maxPos);

            result.setText(classes[maxPos]);
            String s= "";
//            for(int i=0; i<classes.length; i++){
////                s+=String.format("%s: %1.%%\n",classes[i],confidences[i]*100);
//            }

            confidence.setText(s);


            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }


    private void checkPermissionsAndShowDialog() {
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(this, android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
                        != PackageManager.PERMISSION_GRANTED) {
            // Request permissions if not granted
            ActivityCompat.requestPermissions(this,
                    new String[]{android.Manifest.permission.CAMERA, android.Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    REQUEST_CAMERA_PERMISSION);
        } else {
            // Show image source dialog if permissions are granted
            showImageSourceDialog();
        }
    }

    private void showImageSourceDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Select Image Source");
        builder.setItems(new CharSequence[]{"Camera", "Gallery"}, new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                if (which == 0) {
                    openCamera();
                } else {
                    openGallery();
                }
            }
        });
        builder.show();
    }


    private void openCamera() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        } else {
            Toast.makeText(this, "No camera app found", Toast.LENGTH_SHORT).show();
        }
    }

    private void openGallery() {
        Intent pickImageIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(pickImageIntent, REQUEST_IMAGE_PICK);
    }


}