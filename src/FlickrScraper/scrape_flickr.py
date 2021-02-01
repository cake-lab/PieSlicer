#!env python

from BeautifulSoup import BeautifulSoup
import urllib2
import re

import flickrapi
import urllib
from PIL import Image

import numpy as np
from PIL import Image
import os
import cv2
import math

import time
import logging
import hashlib
import random

import uuid

logger = logging.getLogger()

logger.setLevel(logging.INFO)

# TODO: Update API key
API_ID, API_KEY = None, None 

dataset_dir = "raw_dataset"
binned_dataset_dir = "binned_dataset"


dataset_dir = "raw_dataset"
binned_dataset_dir = "binned_dataset"

max_file_size = 10 * (10**6)
num_steps = 100
step_size = 100000
max_bin_size = 10

def getInputFiles(in_dir):
    return [os.path.join(in_dir, f) for f in os.listdir(in_dir) if ".DS_Store" not in f]

def annotateWithFileSize(in_files):
    return [ (os.path.getsize(f), f) for f in in_files ]

def removeLargeFiles(annotated_files):
    return filter( (lambda f: f[0] < max_file_size), annotated_files)

def generateRanges(min_size=0, max_size=max_file_size, step=step_size):
    base_range = range(min_size, max_size+step, step)
    return zip(base_range, base_range[1:])

def filterRange(low_end, high_end, annotated_files):
    in_range = filter( (lambda f: low_end < f[0] and f[0] <= high_end), annotated_files)
    out_range = filter( (lambda f: f not in in_range), annotated_files)
    return in_range, out_range

def formBins():
    
    annotated_files = annotateWithFileSize(getInputFiles(dataset_dir))
    filtered_files = removeLargeFiles(annotated_files)
    for pair in sorted(filtered_files):
        print pair
    size_ranges = generateRanges()
    
    bins = {}
    for low, high in size_ranges:
        in_range, annotated_files = filterRange(low, high, annotated_files)
        bins[ (low, high) ] = in_range
        
        print low, high, len(in_range)
    print len(annotated_files)
    return bins
    
def areBinsFull(bins, full_number=10):
    bin_contents = bins.values()
    full_bins = len( filter( (lambda b: len(b) >= full_number), bin_contents) )
    num_bins = len(bin_contents)
    print "%0.2f%% full" % ( float(full_bins) / num_bins )
    return full_bins == num_bins

def addToBins(bins, local_path):
    file_size = os.path.getsize(local_path)
    bottom = file_size - (file_size % step_size)
    top = bottom + step_size
    index = (bottom, top)
    if len(bins[index]) >= max_bin_size:
        os.remove(local_path)
        return False
    else:
        bins[index].append(local_path)
        return True
    

def getFlickrObj():
    flickr = flickrapi.FlickrAPI(API_ID, API_KEY, cache=True)
    return flickr

def getRandomWords():
    with open('/usr/share/dict/words') as fid:
        words = [s.strip() for s in fid.readlines()]
    random.shuffle(words)
    
    for word in words:
        yield word

def getRemotePhotos(keyword='husky'):
    
    flickr = getFlickrObj()
    
    photos = flickr.walk(text=keyword,
                         tag_mode='all',
                         tags=keyword,
                         extras='url_c',
                         per_page=100,
                         sort='relevance')
    
    
    photo_count = 0
    
    for i, photo in enumerate(photos):
        
        photo_id = photo.get('id')
        sizes_element =  flickr.photos_getSizes(photo_id = photo_id)
        #get an interator
        sizes_iter = sizes_element.iter('size')
        for size in sizes_iter:
            #check if its original size
            if size.attrib["label"] == "Original":
                url = size.attrib["source"]
                yield downloadImage(url)
        if i > 1000:
            break
        #time.sleep(0.25)



def cycleWithAPI(bins, max_samples=100, max_samples_per_word=100):
    i = 0
    for word in getRandomWords():
        print word
        j = 0
        for local_photo in getRemotePhotos(word):
            print local_photo
            
            if areBinsFull(bins):
                return
            
            # Check to see if done overall
            print i
            i += 1
            if i > max_samples:
                return
            
            # Check to see if done with word
            print j
            j += 1
            if j > max_samples_per_word:
                break

def downloadImage(url):
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    filename = url[url.rfind("/")+1:]
    print filename
    local_path = os.path.join(dataset_dir, filename)
    if not os.path.exists(local_path):
        urllib.urlretrieve(url, os.path.join(dataset_dir, filename))
    return local_path

def main():
    
    flickr = getFlickrObj()
    cycleWithAPI(100000, 100)



if __name__ == '__main__':
    main()