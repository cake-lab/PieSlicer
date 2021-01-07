package wpi.ssogden.deeplearningapp;

import android.content.Context;
import android.util.Log;

import com.squareup.okhttp.MediaType;
import com.squareup.okhttp.MultipartBuilder;
import com.squareup.okhttp.OkHttpClient;
import com.squareup.okhttp.Request;
import com.squareup.okhttp.RequestBody;
import com.squareup.okhttp.Response;

import java.io.File;
import java.io.IOException;

import static java.lang.System.nanoTime;

/**
 * Created by samuelogden on 3/20/18.
 */

public class RPCClassifier {

    private static String server_url = "http://35.170.196.199:54321/modipick";

    public static final MediaType MEDIA_TYPE_JPEG = MediaType.parse("image/jpeg");
    private static OkHttpClient client = new OkHttpClient();

    //protected final OkHttpClient client = new OkHttpClient();


    public static Result classify_picture(Context context, String model_name, String mCurrentPhotoPath) {
        Result result = new Result();
        try {
            double startTime_nanos = nanoTime();
            String response_str = doPostRequest(context, mCurrentPhotoPath);
            Log.d("response", response_str);
            String[] results = response_str.split(" ");
            double total_time = nanoTime() - startTime_nanos;
            Log.d("response:", response_str);

            double network_time = (total_time/1000.0/1000.0) - Double.parseDouble(results[2]);


            //result = new Result(results[0], results[1], results[2], String.valueOf(network_time), String.valueOf(total_time/1000.0/1000.0));
            Log.v("total time", String.valueOf(total_time));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return result;
    }

    protected static String doPostRequest(Context context, String mCurrentPhotoPath) throws IOException {

        String filename = "test_picture.png";
        File file = new File(mCurrentPhotoPath);

        //writeBytesToFile(context.getAssets().open(mCurrentPhotoPath), file);

        RequestBody requestBody = new MultipartBuilder()
                .type(MultipartBuilder.FORM)
                .addFormDataPart("file", filename, RequestBody.create(MEDIA_TYPE_JPEG, file))
                .build();

        Request request = new Request.Builder()
                .url(server_url)
                .post(requestBody)
                .build();

        Response response = client.newCall(request).execute();
        return response.body().string();
    }

}
