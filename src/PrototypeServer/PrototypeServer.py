

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import os
import random

import numpy as np

import flask
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename

import logging
from logging import Formatter

import urllib2
from io import BytesIO
import tarfile

#import PIL
from PIL import Image
import json

local_data = "imgs/"
UPLOAD_FOLDER = "imgs/"
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

Image.MAX_IMAGE_PIXELS = None


app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def getTimeInMillis():
    return time.time() * 1000.0

def timeit(method):

    def timed(*args, **kwargs):
        ts = getTimeInMillis()
        result = method(*args, **kwargs)
        te = getTimeInMillis()
        
        return result, (te-ts)

    return timed




def getTime():
    return str(time.time())

@app.route('/ping', methods=['GET'])
def ping_main():
    return "%f" % getTimeInMillis()

@timeit
def saveToDisk(uploaded_file):
    
    filename = secure_filename("temp.jpg")
    local_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    uploaded_file.save(local_filename)
    return local_filename

@timeit
def runpieslicer_internal(save_time, server_prep_time):
    
    t_device_prep = float(request.form["t_device_prep"])
    estimated_transfer_time = float(request.form["estimated_transfer_time"])
    t_sla = float(request.form["t_sla"])

    estimated_network_time = estimated_transfer_time - (save_time)
    
    t_budget = t_sla - (t_device_prep + estimated_transfer_time + server_prep_time + estimated_network_time)
    
    return estimated_network_time, server_prep_time 

@timeit
def loadImage(local_filename, *args, **kwargs):
    img = Image.open(local_filename).convert('RGB')
    return img

@timeit
def resizeImage(img, input_height, input_width, *args, **kwargs):
    img = img.resize((input_width, input_height), Image.ANTIALIAS)
    return img

@timeit
def convertImageToTensor(img):
    img_arr = np.array(img)
    height = img_arr.shape[0]
    width = img_arr.shape[1]
    return img_arr[:, :, 0:3].reshape(1, height, width, 3)

@app.before_request
def before_request_func():
    flask.g.response_start_time = getTimeInMillis()

    
@timeit
def runpieslicer(*args, **kwargs):
    t_msg_arr = flask.g.response_start_time
    t_routing_delay = (getTimeInMillis() - t_msg_arr)

    if False:
        # Save file to disk
        local_filename, save_time = saveToDisk(request.files['file'])
        
        t_transfer_complete = getTimeInMillis()

        # Load Image
        img, load_time = loadImage(local_filename)
    else:
        # Save file to disk
        #local_filename, save_time = saveToDisk(request.files['file'])
        save_time = 0.0
        
        t_transfer_complete = getTimeInMillis()

        # Load Image
        img, load_time = loadImage(request.files['file'])

    # Do general resize
    general_resized_img, general_resize_time = resizeImage(img, max_height, max_width)
    
    # Select a model
    (estimated_network_time, t_server_prep), pieslicer_time = runpieslicer_internal(save_time, (load_time + general_resize_time))
    #t_network = 0.0
    t_transfer = 0.0
    
    # Do model-specific resize
    specific_resized_img, specific_resize_time = resizeImage(general_resized_img, 331, 331)
    
    # Convert image to tensor
    final_img, convert_to_tensor_time = convertImageToTensor(specific_resized_img)

    result, inference_time = "Testing", 0.
    
    info_dict = {
        "routing_time"          : t_routing_delay,
        "save_time"             : save_time,
        "network_time"          : estimated_network_time,
        "transfer_time"         : t_transfer,
        "server_prep_time"      : t_server_prep,
        "pieslicer_time"       : pieslicer_time,
        "load_time"             : load_time,
        "general_resize_time"   : general_resize_time,
        "specific_resize_time"  : specific_resize_time,
        "convert_time"          : convert_to_tensor_time,
        "inference_time"        : inference_time,
        "post_network_time"     : (load_time + general_resize_time + pieslicer_time + specific_resize_time + convert_to_tensor_time + inference_time),
        "model"                 : "test-model",
        "accuracy"              : "0.0",
        "time_budget"           : 0.0,
        "result"                : result,
        
    }
    
    return info_dict
    
    

@app.route('/pieslicer',methods=['POST'])
def pieslicer_main():
    if request.method == 'POST':
        info_dict, total_time = runpieslicer()
        info_dict["total_time"] = total_time
        return json.dumps(info_dict)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--single_model", action="store_true")
    args = parser.parse_args()

    if args.graph:
        model_files = args.graph
    if args.image:
        file_name = args.image
    
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    logging.basicConfig(level=logging.INFO)
    
    with open('times.csv', 'w') as fid:
        fid.write('time_run,model_name,inference_time\n')

    max_height = 331
    max_width = 331
    
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=int("54321"), debug=True, use_reloader=False)
    
    
########################################
