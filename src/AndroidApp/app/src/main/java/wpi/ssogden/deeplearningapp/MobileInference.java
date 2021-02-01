package wpi.ssogden.deeplearningapp;

import android.Manifest;
import android.content.ContentValues;
import android.content.Context;
import android.content.pm.PackageManager;
import android.database.sqlite.SQLiteDatabase;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.CheckBox;
import android.widget.TextView;
import android.widget.ToggleButton;


import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;

import static java.lang.System.nanoTime;
import static wpi.ssogden.deeplearningapp.R.*;

public class MobileInference extends AppCompatActivity{

    private int SLA_STEP = 350;
    private int MAX_SLA = 350;
    private int TIMES_TO_REPEAT = 1;
    private int MAX_TRIES = 5;

    private String device_name = "motox";
    private String network_name = "campus";


    private String[] preprocessedImages = {};

    SQLiteDatabase inference_db;



    double total_misc_time = 0.0;
    double total_load_time = 0.0;
    double total_infer_time = 0.0;


    private class TestModelInBackground extends AsyncTask<String, Float, Long>  {
        //String model_text;
        protected void onPreExecute() {
            // Do nothing right now
        }

        protected Long doInBackground(String... test_files) {

            TextView locationTextview = (TextView) findViewById(id.location_textview);
            String run_tag = locationTextview.getText().toString();

            boolean disable_network = ((ToggleButton)findViewById(id.networkDisable)).isChecked();

            boolean run_static_local = ((CheckBox)findViewById(id.radioStaticLocal)).isChecked();
            boolean run_static_remote = ((CheckBox)findViewById(id.radioStaticRemote)).isChecked();
            boolean run_dynamic = ((CheckBox)findViewById(id.radioDynamic)).isChecked();
            boolean run_dynamic_inverse = ((CheckBox)findViewById(id.radioDynamicInverse)).isChecked();
            boolean do_short_circuit = ((CheckBox)findViewById(id.radioDoShortCircuit)).isChecked();
            boolean add_delta = ((CheckBox)findViewById(id.radioAddDelta)).isChecked();
            boolean run_variations = ((CheckBox)findViewById(id.radioRunVariations)).isChecked();

            int algo_multiplier = 0;
            if (run_static_local) {
                algo_multiplier++;
            }
            if (run_static_remote) {
                algo_multiplier++;
            }
            if (run_dynamic) {
                algo_multiplier++;
            }
            if (run_dynamic_inverse) {
                algo_multiplier++;
            }
            if (run_variations) {
                algo_multiplier += 4;
            }
            int total_tests = TIMES_TO_REPEAT * test_files.length * algo_multiplier;
            int current_progress = 0;

            for (int repeat_counter = 0; repeat_counter < TIMES_TO_REPEAT; repeat_counter++){
                for (int i = 0; i < test_files.length; i++) {

                    Log.d("test file", test_files[i]);

                    if (run_static_local) {
                        Log.d("Algo", "static local");
                        TestModelOnPicture(getApplicationContext(), test_files[i], run_tag, true, false, false, false, false, disable_network, false, add_delta);
                        publishProgress(((float) current_progress++) / total_tests);
                    }
                    if (run_static_remote) {
                        Log.d("Algo", "static remote");
                        TestModelOnPicture(getApplicationContext(), test_files[i], run_tag, false, true, false, false, false, disable_network, false, add_delta);
                        publishProgress(((float) current_progress++) / total_tests);
                    }
                    if (run_dynamic) {
                        Log.d("Algo", "dynamic");
                        TestModelOnPicture(getApplicationContext(), test_files[i], run_tag, false, false, true, false, true, disable_network, do_short_circuit, add_delta);
                        publishProgress(((float) current_progress++) / total_tests);
                    }
                    if (run_dynamic_inverse) {
                        Log.d("Algo", "dynamic inverse");
                        TestModelOnPicture(getApplicationContext(), test_files[i], run_tag, false, false, true, true, true, disable_network, true, add_delta);
                        publishProgress(((float) current_progress++) / total_tests);
                    }
                    if (run_variations) {
                        Log.d("Algo", "dynamic");
                        TestModelOnPicture(getApplicationContext(), test_files[i], run_tag, false, false, true, false, true, disable_network, false, false);
                        TestModelOnPicture(getApplicationContext(), test_files[i], run_tag, false, false, true, false, true, disable_network, true, false);
                        TestModelOnPicture(getApplicationContext(), test_files[i], run_tag, false, false, true, false, true, disable_network, false, true);
                        TestModelOnPicture(getApplicationContext(), test_files[i], run_tag, false, false, true, false, true, disable_network, true, true);
                        publishProgress(((float) current_progress++) / total_tests);
                    }

                }
            }

            Log.v("Load Time", String.valueOf((total_load_time / preprocessedImages.length)));
            Log.v("Infer Time", String.valueOf((total_infer_time / preprocessedImages.length)));
            Log.v("Misc Time", String.valueOf((total_misc_time / preprocessedImages.length)));
            return null;
        }

        protected void onProgressUpdate(Float... progress){
            updateProgress(String.format("%.02f %%", 100.0 * progress[0]));
        }

        protected void onPostExecute(Long result){
            updateProgress("Done!");

        }

    }

    protected void updateProgress(String progress){

        TextView progressText = (TextView)findViewById(id.progressText);
        progressText.setText(progress);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(layout.activity_mobile_inference);

        updateProgress("Ready");

        isStoragePermissionGranted();

        InferenceDbHelper inference_db_helper = new InferenceDbHelper(getApplicationContext());
        inference_db = inference_db_helper.getWritableDatabase();

    }

    public  boolean isStoragePermissionGranted() {
        if (Build.VERSION.SDK_INT >= 23) {
            if (checkSelfPermission(android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    == PackageManager.PERMISSION_GRANTED) {
                Log.v("permissions","Permission is granted");
                return true;
            } else {

                Log.v("permissions","Permission is revoked");
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
                return false;
            }
        }
        else { //permission is automatically granted on sdk<23 upon installation
            Log.v("permissions","Permission is granted");
            return true;
        }
    }

    public void clickGoButton(View view) {

        Log.v("Action", "Go!");

        boolean use_train = ((CheckBox)findViewById(id.radioTraining)).isChecked();
        boolean use_test = ((CheckBox)findViewById(id.radioTesting)).isChecked();

        new TestModelInBackground().execute(getFiles(use_train, use_test));

    }




    protected void TestModelOnPicture(Context context, String img_path, String run_tag, boolean do_static_local, boolean do_static_remote, boolean do_dynamic, boolean invert_model, boolean check_size, boolean disable_network, boolean do_short_circuit, boolean add_delta) {

        int[] prep_dims = new int[] {331};

        for (int i = 0; i < prep_dims.length; i++) {
            for (int sla_goal = SLA_STEP; sla_goal <= MAX_SLA; sla_goal += SLA_STEP) {

                boolean success = false;
                int num_tries = 0;

                while ( ! success && num_tries < MAX_TRIES ) {
                    try {
                        runInference_wrapper(context, img_path, sla_goal, prep_dims[i], run_tag, do_static_local, do_static_remote, do_dynamic, invert_model, check_size, disable_network, do_short_circuit, add_delta);
                        success = true;
                    } catch (JSONException e) {
                        e.printStackTrace();
                        num_tries++;
                    }
                }
            }
        }
    }

    private double nanosToMillis(double nano_time){
        return nano_time / 1000.0 / 1000.0;
    }

    protected void runInference_wrapper(Context context, String img_path, int sla_goal, int prep_dims, String run_tag, boolean do_static_local, boolean do_static_remote, boolean do_dynamic, boolean invert_model, boolean check_size, boolean disable_network, boolean do_short_circuit, boolean add_delta) throws JSONException {

        //RemoteClassifier.updateOffset();

        if ( run_tag.length() != 0) {
            run_tag += " ";
        }
        run_tag += device_name;
        run_tag += " " + network_name;
        if (invert_model) {
            run_tag += " inverted";
        }



        ImageConverter converter = new ImageConverter();
        String preprocessedImage = null;

        double pingTime = RemoteClassifier.getRTT(disable_network);

        ContentValues values = new ContentValues();



        // Start Inference Request Timing!
        double nanos_total_start = nanoTime();

        double nanos_resize_time = 0;
        double nanos_resize_save_time = 0;
        double nanos_modiprep_total = 0;



        int image_x = 0;
        int image_y = 0;

        double nanos_preprocess_start = nanoTime();
        if (do_static_local) {
            // Static Local Code
            values.put(InferenceContract.InferenceEntry.COLUMN_NAME_ALGORITHM, "static local");

            values.put(InferenceContract.InferenceEntry.COLUMN_NAME_PREPROCESS_LOCATION, "local");
            values.put(InferenceContract.InferenceEntry.COLUMN_NAME_PREPROCESS_LOCATION_REAL, "local");

            double nanos_resize_start = nanoTime();
            final BitmapFactory.Options options = new BitmapFactory.Options();
            options.inJustDecodeBounds = true;
            BitmapFactory.decodeFile(img_path, options);
            image_x = options.outWidth;
            image_y = options.outHeight;
            Bitmap resized_bitmap = converter.preprocessPicture_bitmapfactory(img_path, prep_dims, image_x, image_y);
            nanos_resize_time = nanoTime() - nanos_resize_start;

            double nanos_resize_save_start = nanoTime();
            String basename = img_path.substring(img_path.lastIndexOf('/'), img_path.lastIndexOf('.'));
            preprocessedImage = converter.saveBitmapToExternal(resized_bitmap, basename, "preprocessedImages");
            nanos_resize_save_time = nanoTime() - nanos_resize_save_start;

        } else if (do_static_remote) {
            // Static Remote code
            values.put(InferenceContract.InferenceEntry.COLUMN_NAME_ALGORITHM, "static remote");

            values.put(InferenceContract.InferenceEntry.COLUMN_NAME_PREPROCESS_LOCATION, "remote");
            values.put(InferenceContract.InferenceEntry.COLUMN_NAME_PREPROCESS_LOCATION_REAL, "remote");

            preprocessedImage = converter.preprocessPicture_passthrough(img_path);

        } else if (do_dynamic) {
            // Dynamic decision code
            String algo_string = "dynamic";
            if (do_short_circuit) {
                algo_string += " short_circuit";
            }
            if (add_delta) {
                algo_string += " add_delta";
            }
            if (invert_model) {
                algo_string += " invert";
            }
            values.put(InferenceContract.InferenceEntry.COLUMN_NAME_ALGORITHM, algo_string);


            double nanos_preprocess_check_start = nanoTime();
            boolean force_remote = false;

            double nanos_preprocess_check_filesize = 0.0;
            double nanos_preprocess_check_dimensions = 0.0;

            double estimated_preexec_time = 0;

            if (do_short_circuit) {
                if ((new File(img_path)).length() < converter.average_sent_size_in_bytes) {
                    force_remote = true;
                }
                nanos_preprocess_check_filesize = nanoTime() - nanos_preprocess_check_start;
                if (!force_remote) {

                    final BitmapFactory.Options options = new BitmapFactory.Options();
                    options.inJustDecodeBounds = true;
                    BitmapFactory.decodeFile(img_path, options);

                    if (options.outHeight < prep_dims || options.outWidth < prep_dims) {
                        force_remote = true;
                    } else {
                        force_remote = false;
                    }
                    image_x = options.outWidth;
                    image_y = options.outHeight;
                }
                nanos_preprocess_check_dimensions = nanoTime() - (nanos_preprocess_check_start + nanos_preprocess_check_filesize);
            }
            //boolean force_remote = converter.doForceRemoteCheck(img_path, prep_dims);

            double nanos_preprocess_check_total = nanoTime() - nanos_preprocess_check_start;
            values.put(InferenceContract.InferenceEntry.COLUMN_NAME_PREPROCESS_CHECK, nanosToMillis(nanos_preprocess_check_total));
            values.put(InferenceContract.InferenceEntry.COLUMN_NAME_PREPROCESS_CHECK_FILESIZE, nanosToMillis(nanos_preprocess_check_filesize));
            values.put(InferenceContract.InferenceEntry.COLUMN_NAME_PREPROCESS_CHECK_DIMENSIONS, nanosToMillis(nanos_preprocess_check_dimensions));


            if (check_size && force_remote ) {
                // Then do remote by default
                Log.i("Location", "remote");
                values.put(InferenceContract.InferenceEntry.COLUMN_NAME_PREPROCESS_LOCATION, "local");
                values.put(InferenceContract.InferenceEntry.COLUMN_NAME_PREPROCESS_LOCATION_REAL, "remote");
                preprocessedImage = converter.preprocessPicture_passthrough(img_path);
            } else {

                double nanos_modiprep_start = nanoTime();

                long filesize = (new File(img_path)).length();
                double expected_local_prep_time = converter.getLocalEstimate(filesize, device_name, network_name, add_delta);
                double expected_remote_prep_time = converter.getRemoteEstimate(filesize, device_name, network_name, add_delta);
                boolean local_prep_decision = expected_remote_prep_time > expected_local_prep_time;

                nanos_modiprep_total = nanoTime() - nanos_modiprep_start;

                values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_LOCAL_PIESLICER, nanos_modiprep_total);
                values.put(InferenceContract.InferenceEntry.COLUMN_NAME_EXPECTED_TIME_LOCAL_PREP, expected_local_prep_time);
                values.put(InferenceContract.InferenceEntry.COLUMN_NAME_EXPECTED_TIME_REMOTE_PREP, expected_remote_prep_time);

                if (local_prep_decision ^ invert_model) {
                    values.put(InferenceContract.InferenceEntry.COLUMN_NAME_PREEXECUTION_ESTIMATE, expected_local_prep_time);

                    Log.i("Location", "local");
                    values.put(InferenceContract.InferenceEntry.COLUMN_NAME_PREPROCESS_LOCATION, "local");
                    values.put(InferenceContract.InferenceEntry.COLUMN_NAME_PREPROCESS_LOCATION_REAL, "local");

                    double nanos_resize_start = nanoTime();
                    if (image_x == 0 && image_y == 0) {
                        final BitmapFactory.Options options = new BitmapFactory.Options();
                        options.inJustDecodeBounds = true;
                        BitmapFactory.decodeFile(img_path, options);
                        image_x = options.outWidth;
                        image_y = options.outHeight;
                    }
                    Bitmap resized_bitmap = converter.preprocessPicture_bitmapfactory(img_path, prep_dims, image_x, image_y);
                    nanos_resize_time = nanoTime() - nanos_resize_start;

                    double nanos_resize_save_start = nanoTime();
                    String basename = img_path.substring(img_path.lastIndexOf('/'), img_path.lastIndexOf('.'));
                    preprocessedImage = converter.saveBitmapToExternal(resized_bitmap, basename, "preprocessedImages");
                    nanos_resize_save_time = nanoTime() - nanos_resize_save_start;
                } else {
                    values.put(InferenceContract.InferenceEntry.COLUMN_NAME_PREEXECUTION_ESTIMATE, expected_remote_prep_time);

                    Log.i("Location", "remote");
                    values.put(InferenceContract.InferenceEntry.COLUMN_NAME_PREPROCESS_LOCATION, "remote");
                    values.put(InferenceContract.InferenceEntry.COLUMN_NAME_PREPROCESS_LOCATION_REAL, "remote");
                    preprocessedImage = converter.preprocessPicture_passthrough(img_path);
                }
            }


        }
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_ORIG_DIMENSIONS_X, image_x);
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_ORIG_DIMENSIONS_Y, image_y);
        double nanos_preprocess_time = nanoTime() -  nanos_preprocess_start;

        double estimated_transfer_time = converter.getTransferEstimate( (new File(preprocessedImage)).length() , network_name, device_name, add_delta);

        double nanos_local_time = nanos_total_start - nanoTime();

        double nanos_local_remote_start = nanoTime();
        JSONObject remote_results_json = RemoteClassifier.classify_picture(context, "", preprocessedImage, (double) sla_goal, nanosToMillis(nanoTime() - nanos_total_start), estimated_transfer_time, disable_network);
        double nanos_local_remote = nanoTime() - nanos_local_remote_start;

        double nanos_total_time = nanoTime() - nanos_total_start;
        // End Inference request


        double transfer_time = nanosToMillis(nanos_local_remote) - remote_results_json.getDouble("post_network_time");
        double transfer_delta_raw = transfer_time - estimated_transfer_time;
        double transfer_delta = converter.updateNetworkDelta(transfer_time, estimated_transfer_time);



        //double offset_change = RemoteClassifier.updateOffset();

        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TEST_IMAGE_BOOL       , img_path.contains("/test/"));

        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_PING_TIME            , pingTime);

        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_OFFSET_CHANGE            , 0.0);

        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_SLA_TARGET               , sla_goal);
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_NETWORK_OFFSET           , RemoteClassifier.server_offset);

        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TRANSFER_TIME_ESTIMATE    , estimated_transfer_time );
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TRANSFER_TIME_REAL        , transfer_time );

        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TRANSFER_DELTA_RAW            , transfer_delta_raw );
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TRANSFER_DELTA              , transfer_delta );

        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_IMAGENAME_ORIG           , img_path );
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_IMAGENAME_SENT           , preprocessedImage );
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_IMAGESIZE_ORIG           , (new File(img_path)).length() );
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_IMAGESIZE_SENT           , (new File(preprocessedImage)).length() );
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_IMAGE_DIMS               , prep_dims );
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_LOCAL_PIESLICER      , nanosToMillis(nanos_modiprep_total));

        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_LOCAL_PREPROCESS    , nanosToMillis(nanos_preprocess_time ));
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_LOCAL_PREPROCESS_RESIZE    , nanosToMillis(nanos_resize_time ));
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_LOCAL_PREPROCESS_SAVE      , nanosToMillis(nanos_resize_save_time) );

        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_LOCAL_TOTAL    , nanosToMillis(nanos_local_time ));

        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_LOCAL_REMOTE        , nanosToMillis(nanos_local_remote) );


        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_REMOTE_ROUTING         , remote_results_json.getDouble("routing_time") );

        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_REMOTE_SAVE         , remote_results_json.getDouble("save_time") );
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_REMOTE_NETWORK      , remote_results_json.getDouble("network_time") );
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_REMOTE_TRANSFER     , remote_results_json.getDouble("transfer_time") );

        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_REMOTE_PREPIESLICER     , remote_results_json.getDouble("server_prep_time") );

        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_REMOTE_PIESLICER   , remote_results_json.getDouble("pieslicer_time") );
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_REMOTE_LOAD         , remote_results_json.getDouble("load_time") );

        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_REMOTE_GENERAL_RESIZE       , remote_results_json.getDouble("general_resize_time") );
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_REMOTE_SPECIFIC_RESIZE       , remote_results_json.getDouble("specific_resize_time") );
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_REMOTE_CONVERT                 , remote_results_json.getDouble("convert_time") );
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_REMOTE_POST_NETWORK       , remote_results_json.getDouble("post_network_time") );

        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_REMOTE_INFERENCE    , remote_results_json.getDouble("inference_time") );
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_REMOTE_TOTAL        , remote_results_json.getDouble("total_time") );
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_BUDGET              , remote_results_json.getDouble("time_budget") );
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_MODEL_NAME               , remote_results_json.getString("model") );
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_MODEL_ACCURACY           , remote_results_json.getString("accuracy") );
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_INFERENCE_RESULT         , remote_results_json.getString("result") );
        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_TIME_TOTAL               , nanosToMillis(nanos_total_time ));

        values.put(InferenceContract.InferenceEntry.COLUMN_NAME_RUN_TAG                  , run_tag );

        inference_db.insert(InferenceContract.InferenceEntry.TABLE_NAME, "", values);
    }

    private String[] getFiles(boolean include_train, boolean include_test) {
        //String base_dir = "test_images";

        ArrayList<String> file_list = new ArrayList<String>();

        // TODO: Change to update image path location
        String baseDir = Environment.getExternalStorageDirectory().getAbsolutePath() + "/images/";


        if (include_train) {
            File train_directory = new File(baseDir + "/train/");
            File[] train_files = train_directory.listFiles();

            for (int i = 0; i < train_files.length; i++) {
                file_list.add(baseDir + "/train/" + train_files[i].getName());
            }
        }

        if (include_test) {
            File test_directory = new File(baseDir + "/test/");
            File[] test_files = test_directory.listFiles();
            for (int i = 0; i < test_files.length; i++) {
                file_list.add(baseDir + "/test/" + test_files[i].getName());
            }
        }

        Collections.shuffle(file_list);

        return file_list.toArray(new String[0]);
    }



}
