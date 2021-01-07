package wpi.ssogden.deeplearningapp;

import android.provider.BaseColumns;

/**
 * Created by samuelogden on 3/14/18.
 */

public class InferenceContract {
    private InferenceContract() {

    }
    public static class InferenceEntry implements BaseColumns {
        public static final String TABLE_NAME = "inference_results";

        public static final String COLUMN_NAME_ALGORITHM          = "algorithm";
        public static final String COLUMN_NAME_TEST_IMAGE_BOOL          = "test_image_bool";

        public static final String COLUMN_NAME_SLA_TARGET               = "sla_target";

        public static final String COLUMN_NAME_PREPROCESS_CHECK      = "time_local_preprocess_check";
        public static final String COLUMN_NAME_PREPROCESS_CHECK_FILESIZE      = "time_local_preprocess_check_filesize";
        public static final String COLUMN_NAME_PREPROCESS_CHECK_DIMENSIONS      = "time_local_preprocess_check_dimensions";


        public static final String COLUMN_NAME_PREPROCESS_LOCATION      = "preprocess_location";
        public static final String COLUMN_NAME_PREPROCESS_LOCATION_REAL      = "preprocess_location_real";
        public static final String COLUMN_NAME_RUN_TAG                  = "run_tag";
        public static final String COLUMN_NAME_NETWORK_OFFSET           = "network_offset";

        public static final String COLUMN_NAME_PING_TIME                = "ping_time";

        public static final String COLUMN_NAME_PREEXECUTION_ESTIMATE    = "preexecution_time_estimate";

        public static final String COLUMN_NAME_TRANSFER_TIME_ESTIMATE    = "transfer_time_estimate";
        public static final String COLUMN_NAME_TRANSFER_TIME_REAL        = "transfer_time_real";
        public static final String COLUMN_NAME_TRANSFER_DELTA            = "transfer_time_delta";
        public static final String COLUMN_NAME_TRANSFER_DELTA_RAW        = "transfer_time_delta_raw";

        public static final String COLUMN_NAME_IMAGENAME_ORIG           = "image_name_orig";
        public static final String COLUMN_NAME_IMAGENAME_SENT           = "image_name_sent";
        public static final String COLUMN_NAME_IMAGESIZE_ORIG           = "orig_size";
        public static final String COLUMN_NAME_IMAGESIZE_SENT           = "sent_size";
        public static final String COLUMN_NAME_IMAGE_DIMS               = "image_dims";
        public static final String COLUMN_NAME_ORIG_DIMENSIONS_X      = "orig_dims_x";
        public static final String COLUMN_NAME_ORIG_DIMENSIONS_Y      = "orig_dims_y";

        public static final String COLUMN_NAME_TIME_LOCAL_PIESLICER     = "time_local_pieslicer";

        public static final String COLUMN_NAME_TIME_LOCAL_PREPROCESS    = "time_local_preprocess";
        public static final String COLUMN_NAME_TIME_LOCAL_PREPROCESS_RESIZE    = "time_local_preprocess_resize";
        public static final String COLUMN_NAME_TIME_LOCAL_PREPROCESS_SAVE    = "time_local_preprocess_save";

        public static final String COLUMN_NAME_TIME_LOCAL_TOTAL        = "time_local_total";

        public static final String COLUMN_NAME_TIME_LOCAL_REMOTE        = "time_local_remote";

        public static final String COLUMN_NAME_EXPECTED_TIME_LOCAL_PREP        = "expected_time_local_prep";
        public static final String COLUMN_NAME_EXPECTED_TIME_REMOTE_PREP       = "expected_time_remote_prep";


        public static final String COLUMN_NAME_TIME_REMOTE_ROUTING         = "time_remote_routing";

        public static final String COLUMN_NAME_TIME_REMOTE_SAVE         = "time_remote_save";
        public static final String COLUMN_NAME_TIME_REMOTE_NETWORK      = "time_remote_network";
        public static final String COLUMN_NAME_TIME_REMOTE_TRANSFER     = "time_remote_transfer";
        public static final String COLUMN_NAME_TIME_REMOTE_PREPIESLICER     = "time_remote_prepieslicer";

        public static final String COLUMN_NAME_TIME_REMOTE_PIESLICER   = "time_remote_pieslicer";
        public static final String COLUMN_NAME_TIME_REMOTE_LOAD         = "time_remote_load";
        public static final String COLUMN_NAME_TIME_REMOTE_GENERAL_RESIZE    = "time_remote_general_resize";
        public static final String COLUMN_NAME_TIME_REMOTE_SPECIFIC_RESIZE   = "time_remote_specific_resize";
        public static final String COLUMN_NAME_TIME_REMOTE_CONVERT    = "time_remote_convert";
        public static final String COLUMN_NAME_TIME_REMOTE_POST_NETWORK   = "time_remote_post_network";
        public static final String COLUMN_NAME_TIME_REMOTE_INFERENCE    = "time_remote_inference";
        public static final String COLUMN_NAME_TIME_REMOTE_TOTAL        = "time_remote_total";

        public static final String COLUMN_NAME_TIME_BUDGET              = "time_budget";

        public static final String COLUMN_NAME_MODEL_NAME               = "model_name";
        public static final String COLUMN_NAME_MODEL_ACCURACY           = "model_accuracy";
        public static final String COLUMN_NAME_INFERENCE_RESULT         = "inference_result";

        public static final String COLUMN_NAME_OFFSET_CHANGE        = "offset_change";

        public static final String COLUMN_NAME_TIME_TOTAL               = "time_total";

        public static final String COLUMN_TIME_STAMP = "timeStamp";


    }

    static final String SQL_CREATE_ENTRIES =
            "CREATE TABLE " + InferenceEntry.TABLE_NAME + " (" +
                    InferenceEntry._ID + " INTEGER PRIMARY KEY," +

                    InferenceEntry.COLUMN_NAME_ALGORITHM           + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_TEST_IMAGE_BOOL           + " TEXT DEFAULT ''," +

                    InferenceEntry.COLUMN_NAME_SLA_TARGET               + " TEXT DEFAULT ''," +

                    InferenceEntry.COLUMN_NAME_PREPROCESS_CHECK               + " TEXT DEFAULT '0'," +
                    InferenceEntry.COLUMN_NAME_PREPROCESS_CHECK_FILESIZE               + " TEXT DEFAULT '0'," +
                    InferenceEntry.COLUMN_NAME_PREPROCESS_CHECK_DIMENSIONS               + " TEXT DEFAULT '0'," +

                    InferenceEntry.COLUMN_NAME_RUN_TAG                  + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_NETWORK_OFFSET           + " TEXT DEFAULT ''," +

                    InferenceEntry.COLUMN_NAME_PING_TIME           + " TEXT DEFAULT ''," +

                    InferenceEntry.COLUMN_NAME_PREEXECUTION_ESTIMATE    + " TEXT DEFAULT '0'," +

                    InferenceEntry.COLUMN_NAME_TRANSFER_TIME_ESTIMATE    + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_TRANSFER_TIME_REAL        + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_TRANSFER_DELTA            + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_TRANSFER_DELTA_RAW        + " TEXT DEFAULT ''," +

                    InferenceEntry.COLUMN_NAME_IMAGENAME_ORIG           + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_IMAGENAME_SENT           + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_IMAGESIZE_ORIG           + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_IMAGESIZE_SENT           + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_IMAGE_DIMS               + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_ORIG_DIMENSIONS_X               + " TEXT DEFAULT '0'," +
                    InferenceEntry.COLUMN_NAME_ORIG_DIMENSIONS_Y               + " TEXT DEFAULT '0'," +

                    InferenceEntry.COLUMN_NAME_TIME_LOCAL_PIESLICER      + " TEXT DEFAULT ''," +

                    InferenceEntry.COLUMN_NAME_TIME_LOCAL_PREPROCESS    + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_TIME_LOCAL_PREPROCESS_RESIZE    + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_TIME_LOCAL_PREPROCESS_SAVE    + " TEXT DEFAULT ''," +

                    InferenceEntry.COLUMN_NAME_TIME_LOCAL_TOTAL         + " TEXT DEFAULT ''," +



                    InferenceEntry.COLUMN_NAME_TIME_LOCAL_REMOTE        + " TEXT DEFAULT ''," +

                    InferenceEntry.COLUMN_NAME_EXPECTED_TIME_LOCAL_PREP        + " TEXT DEFAULT '0'," +
                    InferenceEntry.COLUMN_NAME_EXPECTED_TIME_REMOTE_PREP       + " TEXT DEFAULT '0'," +

                    InferenceEntry.COLUMN_NAME_PREPROCESS_LOCATION      + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_PREPROCESS_LOCATION_REAL      + " TEXT DEFAULT ''," +

                    InferenceEntry.COLUMN_NAME_TIME_REMOTE_ROUTING         + " TEXT DEFAULT ''," +

                    InferenceEntry.COLUMN_NAME_TIME_REMOTE_SAVE         + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_TIME_REMOTE_NETWORK      + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_TIME_REMOTE_TRANSFER     + " TEXT DEFAULT ''," +

                    InferenceEntry.COLUMN_NAME_TIME_REMOTE_PREPIESLICER     + " TEXT DEFAULT ''," +

                    InferenceEntry.COLUMN_NAME_TIME_REMOTE_PIESLICER   + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_TIME_REMOTE_LOAD         + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_TIME_REMOTE_GENERAL_RESIZE       + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_TIME_REMOTE_SPECIFIC_RESIZE       + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_TIME_REMOTE_CONVERT       + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_TIME_REMOTE_POST_NETWORK       + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_TIME_REMOTE_INFERENCE    + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_TIME_REMOTE_TOTAL        + " TEXT DEFAULT ''," +

                    InferenceEntry.COLUMN_NAME_TIME_BUDGET        + " TEXT DEFAULT ''," +

                    InferenceEntry.COLUMN_NAME_MODEL_NAME               + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_MODEL_ACCURACY           + " TEXT DEFAULT ''," +
                    InferenceEntry.COLUMN_NAME_INFERENCE_RESULT         + " TEXT DEFAULT ''," +

                    InferenceEntry.COLUMN_NAME_OFFSET_CHANGE         + " TEXT DEFAULT ''," +

                    InferenceEntry.COLUMN_NAME_TIME_TOTAL         + " TEXT DEFAULT ''," +

                    InferenceEntry.COLUMN_TIME_STAMP + " TIMESTAMP DEFAULT CURRENT_TIMESTAMP );";

    static final String SQL_DELETE_ENTRIES =
            "DROP TABLE IF EXISTS " + InferenceEntry.TABLE_NAME;
}
