#!env python

import os
from PIL import Image
import tempfile
import time

log_file = "JpegQualityMeasurements.csv"
log_fid = open(log_file, 'w')
log_fid.write("timestamp,dataset,image_name,quality,file_size\n")

def recordMeasurement(dataset, image_name, quality, file_size):
    log_fid.write(','.join([str(time.time()), str(dataset), str(image_name), str(quality), str(file_size)]) + '\n')

def getImageFiles(dir):
    image_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if "DS_Store" not in file:
                image_files.append( (root.replace(dir, ""), os.path.join(root, file)) )
    return image_files

def getImageObject(in_image, dims=331):
    image = Image.open(in_image).convert('RGB')
    image_obj = image.resize((dims, dims), Image.ANTIALIAS)
    
    return image_obj

def getImageSizeAtQuality(in_image_obj, quality):
    tmpfile = tempfile.TemporaryFile()
    
    in_image_obj.save(tmpfile, 'JPEG', quality=quality)
    return os.stat(tmpfile.name).st_size
    

def main():
    base_dir = "PATH_TO_IMAGES" # TODO
    
    images = getImageFiles(base_dir)
    print(len(images))
    
    for i, (dataset, image) in enumerate(images):
        print("%0.2f%%" % (100. * i / len(images)))
        file_to_test = image
        try:
            image_obj = getImageObject(file_to_test)
        except OSError:
            continue
        for quality in range(1, 101):
            size_at_quality = getImageSizeAtQuality(image_obj, quality)
            recordMeasurement(dataset, os.path.basename(image), quality, size_at_quality)
            
if __name__ == '__main__':
    main()