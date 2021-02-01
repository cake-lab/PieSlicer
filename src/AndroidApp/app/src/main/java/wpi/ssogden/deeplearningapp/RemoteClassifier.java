package wpi.ssogden.deeplearningapp;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.IOException;


import com.squareup.okhttp.MediaType;
import com.squareup.okhttp.MultipartBuilder;
import com.squareup.okhttp.OkHttpClient;
import com.squareup.okhttp.Request;
import com.squareup.okhttp.RequestBody;
import com.squareup.okhttp.Response;

import org.json.JSONException;
import org.json.JSONObject;

import static java.lang.System.nanoTime;

/**
 * Created by samuelogden on 3/20/18.
 */

public class RemoteClassifier {
    private static String ip_address = "localhost"; // TODO: Change to match your actual server
    private static String server_url = "http://" + ip_address + ":54321/pieslicer";

    public static final MediaType MEDIA_TYPE_JPEG = MediaType.parse("image/jpeg");
    private static OkHttpClient client = new OkHttpClient();

    public static double server_offset = 0.0;
    //protected final OkHttpClient client = new OkHttpClient();


    public static JSONObject classify_picture(Context context, String model_name, String mCurrentPhotoPath, Double sla_goal, Double local_prep_time,  Double estimated_transfer_time, boolean disable_network) throws JSONException {
        if (disable_network) {
            return new JSONObject("" +
                            "{" +
                            "   \"post_network_time\":      0.0, " +
                            "   \"model\":                  \"fake\", " +
                            "   \"server_prep_time\"        :0.0, " +
                            "   \"inference_time\":         0.0, " +
                            "   \"routing_time\":           0.0, " +
                            "   \"time_budget\":            0.0, " +
                            "   \"specific_resize_time\":   0.0, " +
                            "   \"convert_time\":           0.0, " +
                            "   \"general_resize_time\":    0.0, " +
                            "   \"pieslicer_time\":        0.0, " +
                            "   \"result\":                 -1, " +
                            "   \"total_time\":             0.0, " +
                            "   \"load_time\":              0.0, " +
                            "   \"network_time\":           0.0, " +
                            "   \"save_time\":              0.0, " +
                            "   \"transfer_time\":          0.0, " +
                            "   \"accuracy\":               0.0" +
                            "}"
                    );
        }

        JSONObject result = new JSONObject("{}");
        Log.d("image_name", mCurrentPhotoPath);
        try {
            double startTime_nanos = nanoTime();

            String post_url = "http://" + ip_address + ":54321/";
            post_url += "modipick";

            String response_str = doPostRequest(context, post_url, mCurrentPhotoPath, sla_goal, local_prep_time, estimated_transfer_time);

            Log.d("response", response_str);
            String[] results = response_str.split(" ");
            double total_time_nanos = nanoTime() - startTime_nanos;
            Log.d("response:", response_str);

            if ( ! response_str.startsWith("<!DOCTYPE")) {

                result = new JSONObject(response_str);
            } else {
                return result;
            }
        } catch (IOException e) {
            e.printStackTrace();
        } catch (JSONException e) {
            e.printStackTrace();
        }
        return result;
    }

    protected static String doPostRequest(Context context, String server_url, String mCurrentPhotoPath, Double sla_goal, Double local_prep_time,  Double estimated_transfer_time) throws IOException {


        String filename = "test_picture.png";
        File file = new File(mCurrentPhotoPath);

        //writeBytesToFile(context.getAssets().open(mCurrentPhotoPath), file);

        RequestBody requestBody = new MultipartBuilder()
                .type(MultipartBuilder.FORM)
                .addFormDataPart("t_sla", String.valueOf(sla_goal))
                .addFormDataPart("t_device_prep", String.valueOf(local_prep_time))
                .addFormDataPart("estimated_transfer_time", String.valueOf(estimated_transfer_time))

                .addFormDataPart("file", filename, RequestBody.create(MEDIA_TYPE_JPEG, file))
                .build();

        Request request = new Request.Builder()
                .url(server_url)
                .post(requestBody)
                .build();

        Response response = client.newCall(request).execute();
        return response.body().string();
    }



    public static double updateOffset(){


        String hosturl = "http://" + ip_address + ":54321/ping";

        Request request = new Request.Builder()
                .url(hosturl)
                .build();

        Response response = null;

        double old_offset = server_offset;

        try {

            long startTime = System.currentTimeMillis();
            response = client.newCall(request).execute();
            double server_time = Double.parseDouble(response.body().string());
            long rtt = System.currentTimeMillis() - startTime;

            double new_t_local = server_time + (rtt / 2);

            server_offset = System.currentTimeMillis() - new_t_local;

        } catch (IOException e) {
            e.printStackTrace();
        }
        return (old_offset - server_offset);
    }

    public static double getRTT(boolean disable_network){

        if (disable_network){
            return 0.0;
        }

        String hosturl = "http://" + ip_address + ":54321/ping";

        Request request = new Request.Builder()
                .url(hosturl)
                .build();

        double rtt = (long) 0.0;

        try {

            long startTime = System.currentTimeMillis();
            client.newCall(request).execute();
            rtt = System.currentTimeMillis() - startTime;


        } catch (IOException e) {
            e.printStackTrace();
        }
        return rtt;
    }

}
