from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import pandas as pd
from pathlib import Path #added new - Kanav


## kanav added below for CPU run ##
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
boxdim = []

## kanav modification ends ##


detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.2,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=180,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    parser.add_argument(
        '-p-path',
        '--output_path',
        dest='output_path',
        type=str,
        default= "C:/Users/Kanav/Documents/Dissertation/Parkinsons_Disease/Codes/Boundingbox_dim_files",
        help='Provide path of the output bounding box dim.')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_source)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 1
    
    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    while num_frames<=round(cap.get(5)*10): # kanav - keeping hard frame count as problem with cv2. Unable to recognise end of frames
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        # image_np = cv2.flip(image_np, 1)
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

        boxes, scores = detector_utils.detect_objects(image_np,
                                                      detection_graph, sess)
        
        #providing bounding box info
        temp = detector_utils.bounding_box_dim(num_hands_detect, args.score_thresh, 
                                  scores, boxes, im_width, im_height, 
                                  image_np)
        boxdim.append(temp)


        # draw bounding boxes on frame
        detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
                                         scores, boxes, im_width, im_height,
                                         image_np)

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                 image_np)

            cv2.imshow('Single-Threaded Detection',
                       cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("frames processed: ", num_frames, "elapsed time: ",
                  elapsed_time, "fps: ", str(int(fps)))

    #flattening the list[[]] to [] and saving as csv
    boxdim_df = pd.DataFrame([k[0] for k in boxdim], columns= ["filename", "index", "xmin","ymin", "xmax", "ymax"])
    filename = "test_file" + ".csv" #kanav make this dynamic
    boxdim_df.to_csv(Path(args.output_path,filename), index = False)