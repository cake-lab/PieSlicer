package wpi.ssogden.deeplearningapp;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Environment;


import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * Created by samuelogden on 4/22/19.
 */


public class ImageConverter {

    private static double PREPED_SIZE = 0.127;
    public double transfer_delta;
    private static double alpha = 0.5;

    public long average_sent_size_in_bytes = 53982;

    public ImageConverter() {
        transfer_delta = 0.0;
    }

    public double updateNetworkDelta(double network_time, double estimate){

        double old_network_delta = transfer_delta;

        transfer_delta = transfer_delta + (alpha * (network_time - estimate) + (1-alpha) * old_network_delta);

        return transfer_delta;
    }

    public boolean doForceRemoteCheck(String in_jpeg, int prep_dims){

        if ((new File(in_jpeg)).length() < average_sent_size_in_bytes) {
            return true;
        }

        final BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(in_jpeg, options);

        if ( options.outHeight < prep_dims || options.outWidth < prep_dims ) {
            return true;
        }

        return false;

    }

    public Bitmap preprocessPicture_bitmapfactory(String in_jpeg, int prep_dims, int image_x, int image_y) {

        String processed = "";

        // First decode with inJustDecodeBounds=true to check dimensions
        final BitmapFactory.Options options = new BitmapFactory.Options();
        //options.inJustDecodeBounds = true;
        //BitmapFactory.decodeFile(in_jpeg, options);

        //Log.d("Prep_dims", String.valueOf(prep_dims));

        // Calculate inSampleSize
        options.inSampleSize = calculateInSampleSize(image_x, image_y, prep_dims, prep_dims);

        // Decode bitmap with inSampleSize set
        options.inJustDecodeBounds = false;
        Bitmap resized_bitmap = null;
        resized_bitmap = BitmapFactory.decodeFile(in_jpeg, options);
        return resized_bitmap;
    }


    public String preprocessPicture_passthrough(String in_jpeg) {
        /*
        String processed = "";
        String basename = in_jpeg.substring(in_jpeg.lastIndexOf('/'), in_jpeg.lastIndexOf('.')); //FilenameUtils.removeExtension(jpeg_file);

        Bitmap bMap = null;
        bMap = BitmapFactory.decodeFile(in_jpeg);
        processed = saveBitmapToExternal(bMap, basename);
        */
        return in_jpeg;
    }


    public String saveBitmapToExternal(Bitmap finalBitmap, String filename_base, String folder_name) {

        String root = Environment.getExternalStorageDirectory().toString();
        File myDir = new File(root + "/" + folder_name);
        myDir.mkdirs();
        String fname = myDir + filename_base + ".jpg";
        File file = new File(fname);
        if (file.exists())
            file.delete();
        try {
            FileOutputStream out = new FileOutputStream(file);
            finalBitmap.compress(Bitmap.CompressFormat.JPEG, 90, out);
            out.flush();
            out.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
        //Log.v("Filesize:", String.valueOf(file.length()));
        return fname;
    }

    private static int calculateInSampleSize(int image_x, int image_y, int reqWidth, int reqHeight) {
        // Raw height and width of image
        //final int height = options.outHeight;
        //final int width = options.outWidth;
        int inSampleSize = 1;

        if (image_y > reqHeight || image_x > reqWidth) {

            final int halfHeight = image_y / 2;
            final int halfWidth = image_x / 2;

            // Calculate the largest inSampleSize value that is a power of 2 and keeps both
            // height and width larger than the requested height and width.
            while ((halfHeight / inSampleSize) >= reqHeight
                    && (halfWidth / inSampleSize) >= reqWidth) {
                inSampleSize *= 2;
            }
        }

        return inSampleSize;
    }

    /*
    public boolean doLocalPreprocess(String filepath, String device_name, String network_name){
        long filesize = (new File(filepath)).length();

        double f = local_preprocess_time(filesize, device_name);
        double g_prime = network_transfer_time((long) PREPED_SIZE, network_name);
        double h_prime = remote_preprocess_time((long) PREPED_SIZE);

        double g = network_transfer_time(filesize, network_name);
        double h = remote_preprocess_time(filesize);

        return ( (f + g_prime + h_prime) <= (g + h) );
    }
    */

    public double getTransferEstimate(long filesize, String network_name, String device_name, boolean add_delta){
        double filesize_in_MB = filesize / 1000.0 / 1000.0;

        return network_transfer_time(filesize_in_MB, network_name, device_name, add_delta);
    }


    public double getLocalEstimate(long filesize, String device_name, String network_name, boolean add_delta){
        double filesize_in_MB = filesize / 1000.0 / 1000.0;
        double f = local_preprocess_time(filesize_in_MB, device_name);
        double g_prime = network_transfer_time((long) PREPED_SIZE, network_name, device_name, add_delta);
        double h_prime = remote_preprocess_time((long) PREPED_SIZE);

        return (f + g_prime + h_prime);
    }
    public double getRemoteEstimate(long filesize, String device_name, String network_name, boolean add_delta){

        double filesize_in_MB = filesize / 1000.0 / 1000.0;
        double g = network_transfer_time(filesize_in_MB, network_name, device_name, add_delta);
        double h = remote_preprocess_time(filesize_in_MB);

        return (g + h);
    }

    private double local_preprocess_time(double filesize, String device_name){
        double a = 0.0;
        double b = 0.0;

        if (device_name.contains("nexus")) {
            a = 103.982;
            b = 36.018;
        }

        if (device_name.contains("motox")) {
            a = 68.355;
            b = 73.460;
        }

        if (device_name.contains("pixel")) {
            a = 39.143;
            b = 63.091;
        }

        return a * filesize + b;
    }

    private double network_transfer_time(double filesize, String network_name, String device_name, boolean add_delta){
        double a = 0.0; //97.800;
        double b = 0.0; // 125.170;

        if (network_name.contains("campus")) {
            if (device_name.contains("nexus")) {
                a = 389.698;
                b = 104.983;
            }

            if (device_name.contains("motox")) {
                a = 153.665;
                b = 93.489;
            }

            if (device_name.contains("pixel")) {
                a = 94.797;
                b = 98.585;
            }
        } else if (network_name.contains("home")) {

            if (device_name.contains("nexus")) {
                a = 1124.668;
                b = 86.308;
            }

            if (device_name.contains("motox")) {
                a = 1086.079;
                b = 108.006;
            }

            if (device_name.contains("pixel")) {
                a = 1529.267;
                b = 112.212;
            }
        }


        if (a == 0.0 && b == 0.0) {
            return network_transfer_time(filesize, "campus", device_name, add_delta);
        }
        if (add_delta) {
            return a * filesize + b + transfer_delta;
        } else {
            return a * filesize + b;
        }

    }

    private double remote_preprocess_time(double filesize){
        double a = 58.049;
        double b = 3.824;
        return a * filesize + b;
    }
}
