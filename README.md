# PD_handtracking
This repo documents steps and scripts used to train a hand detector using Tensorflow Object Detection API. The original dataset, being confidential, was not uploaded in this repository directory. You can use your own dataset.

## Requirements and environment 
- Installation can of TFOD can be done by following the guide on installing [Tensorflow and the Tensorflow object detection api](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). This will walk you through setting up the tensorflow framework, cloning the tensorflow github repo and a guide on  
- Environment configuration is provided in the requirements_handtrack.txt file



## Using the Detector to Detect/Track hands

- Load the `frozen_inference_graph.pb` trained on the hands dataset as well as the corresponding label map. In this repo, this is done in the `utils/detector_utils.py` script by the `load_inference_graph` method.
  ```python
  detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Hand Inference graph loaded.")
  ```
- Detect hands. In this repo, this is done in the `utils/detector_utils.py` script by the `detect_objects` method.
  ```python
  (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
  ```
- Visualize detected bounding detection_boxes. In this repo, this is done in the `utils/detector_utils.py` script by the `draw_box_on_image` method.

## Scripts for detection of hands  
- (Single file) __myenv_single_threaded.py__ : Main script for reading camera video input detection and detecting. Takes a set of command line flags to set parameters such as `--display` (visualize detections), image parameters `--width` and `--height`, videe `--source` (0 for camera) etc.
- (Multiple files) __script_for_all_bounding_box_coordinates.py__ : If you have all the files in a directory, you can update and run the below script for generating the bounding box coordinates for each frame across all videos. 

-------------------- Adjustment of bounding boxes and video cropping --------------

#### Bounding box adjustments folder contains important codes for adjusting the bounding boxes and then cropping the videos as per the bounding boxes.
 - csv files : files used across different codes.
 - __script_to_find_largest_bounding_box.ipynb__ : Finding the box of max area (post outlier treatment)
 - __script_tocheck_adjusted_labels.py__ : Plotting coordinates on video to check if coordinates adjusted properly
 - __script_transforming_labeled_data.ipynb__ : Script for combining all the manually annotated CSV files
 - __script_for_all_video_crop.py__ : Script to crop the videos as per the adjusted bounding boxes
