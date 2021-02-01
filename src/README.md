# PieSlicer Source Code

This folder containers the source code for enabling the PieSlicer deep learning measurement application, as well as code for a number of subexperiments.

## Contents

- [AndroidApp](#androidapp)
- [PrototypeServer](#prototypeserver)
- [JpegQualityMeasurements](#jpegqualitymeasurements)
- [Modeling](#modeling)
- [FlickrScraper](#flickrscraper)

## Deep Learning Measurement

Two of the included sources, those for the AndroidApp and the Prototype server, comprise the core of PieSlicer.
These applications are to be run in tandem to collect the data used in the paper.

### AndroidApp

The android application requires Android API >= 27 and was tested to run with Android Studio 4.1.2.
It expects the server to be running remotely, which accepts POST requests containing JPEG image data.

To run simply open Android Studio and select "Load Project..." and select the `PieSlicer/src/AndroidApp` directory.

Note, `RemoteClassifier.java` (located at `PieSlicer/src/AndroidApp/app/src/main/java/wpi/ssogden/deeplearningapp/RemoteClassifier.java`) will need to be updated with the appropriate server address.

Images are assumed to be stored locally on the mobile device in external storage at `/images/`.
There are two subdirectories, `test` and `train` that are used to further subdivide and can be enabled or disabled as needed.

After testing the resulting Sqlite3 database will need to be pulled from the android device, likely using the `adb shell` command.

### PrototypeServer

The prototype server was design to use python2.7 (compatibility with python3 not yet tested).
The requirements for the server are found at `src/PrototypeServer/requirements.txt`.

To run the server install the requirements and run `python2.7 PrototypeServer.py`.

## Subexperiments

### Modeling

To test the variations on modeling used in PieSlicer we generated a range of models using [scikit-learn](https://scikit-learn.org/).
To test these the path to the android databases collected above should be entered into the code and allowed to run.

Note: individual models can be turned on and off via the True-False tree starting at line 246.

### JpegQualityMeasurements

The goal of this script is to find the size of image files saved under different JPEG qualities.
Prior to running update the path to the images on line 37.

## Dataset collection

### FlickrScraper

To calculate the distribution of the datasets used run `python scrape_flickr.py`.
Prior to running you must generate your own [Flickr API key](https://www.flickr.com/services/api/).
