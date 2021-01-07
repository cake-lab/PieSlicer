package wpi.ssogden.deeplearningapp;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.SystemClock;
import android.util.Log;

import java.io.IOException;
import java.util.List;

import static java.lang.System.nanoTime;

/**
 * Created by samuelogden on 3/19/18.
 */

public class LocalClassifier {
    private static Classifier classifier;

    public static Result classify_picture(Context context, String model_name, String mCurrentPhotoPath) {

        int input_size = 299;
        String output_name = "MobilenetV1/Predictions/Reshape_1";

        if (model_name.contains("224")) {
            input_size = 224;
            output_name = "MobilenetV1/Predictions/Reshape_1";
        } else if (model_name.contains("224")) {
            input_size = 128;
            output_name = "MobilenetV1/Predictions/Reshape_1";
        } else if (model_name.contains("inception_v3")) {
            input_size = 299;
            output_name = "InceptionV3/Predictions/Reshape_1";
        } else if (model_name.contains("inception_v4")) {
            input_size = 299;
            output_name = "InceptionV4/Logits/Predictions";
        } else if (model_name.contains("nasnet_large")) {
            input_size = 331;
            output_name = "final_layer/predictions";
        }

        return run_classifier(context, mCurrentPhotoPath,
                "file:///android_asset/" + model_name,
                "file:///android_asset/models/labels.txt",
                input_size,
                0,
                255.0f,
                "input",
                output_name);
    }

    public static Result run_classifier(Context context, String mCurrentPhotoPath, String model_file, String label_file, int input_size, int image_mean, float image_std, String input_name, String output_name) {

        long nano_startTotalTime = nanoTime(); // System.currentTimeMillis();

        //Init Classifier
        long nano_startTime_loading = nanoTime(); //System.currentTimeMillis();
        classifier =
                TensorFlowImageClassifier.create(context.getAssets(), model_file, label_file, input_size, image_mean, image_std, input_name, output_name);
        long nano_loadingDuration = System.nanoTime() - nano_startTime_loading;

        // Preprocess image
        long nano_startPreprocess = nanoTime(); //System.currentTimeMillis();

        Bitmap bMap = BitmapFactory.decodeFile(mCurrentPhotoPath); //context.getAssets().open(in_jpeg));

        long nano_preprocessDuration = nanoTime() - nano_startPreprocess;

        // Run inference
        long nano_startTime_inference = nanoTime(); //System.currentTimeMillis();
        final List<Classifier.Recognition> results = classifier.recognizeImage(bMap);
        long nano_inferenceDuration = nanoTime() - nano_startTime_inference;

        double nanos_total_time = nanoTime() - nano_startTotalTime;
        Log.v("load time", String.valueOf(nano_loadingDuration));
        Log.v("infer time", String.valueOf(nano_inferenceDuration));
        Log.v("total time", String.valueOf(nanos_total_time));

        return new Result();
        //return new Result("nothing", (float) .99, (float) (nano_loadingDuration / 1000.0 / 1000.0), (float) (nano_preprocessDuration / 1000.0 / 1000.0), (float) (nano_inferenceDuration / 1000.0 / 1000.0), (float) (nanos_total_time / 1000.0 / 1000.0), model_file);
        //return new Result(results.get(0).getTitle(), results.get(0).getConfidence(), loadingDuration, preprocessDuration, inferenceDuration, (float) total_time, model_file);
    }
}
