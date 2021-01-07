package wpi.ssogden.deeplearningapp;

public class Result {
    public final String save_time;
    public final String preprocess_time;
    public final String read_time;
    public final String resize_time;
    public final String infer_time;
    public final String remotetotal_time;
    public final String cloud_model_name;
    public final String network_transfer_time;

    public Result(String save_time,
                  String preprocess_time,
                  String read_time,
                  String resize_time,
                  String infer_time,
                  String remotetotal_time,
                  String cloud_model_name,
                  String network_transfer_time) {

        this.save_time = save_time;
        this.preprocess_time = preprocess_time;
        this.read_time = read_time;
        this.resize_time = resize_time;
        this.infer_time = infer_time;
        this.remotetotal_time = remotetotal_time;
        this.cloud_model_name = cloud_model_name;
        this.network_transfer_time = network_transfer_time;

        //this.save_time = save_time;
        //this.preprocess_time = preprocess_time;
        //this.total_remote_time = total_remote_time;
        //this.total_time =  total_time;
        //this.network_time = network_time;
    }

    public Result() {
        this.save_time              = "";
        this.preprocess_time        = "";
        this.read_time              = "";
        this.resize_time            = "";
        this.infer_time             = "";
        this.remotetotal_time       = "";
        this.cloud_model_name       = "";
        this.network_transfer_time  = "";
    }
}
