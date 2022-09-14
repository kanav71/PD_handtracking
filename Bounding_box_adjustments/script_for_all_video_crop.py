import numpy as np
import cv2
import tensorflow as tf
import os
from pathlib import Path
import pandas as pd

video_source = "C:/Users/Kanav/Documents/Dissertation/Parkinsons_Disease/Data/parkinsons_videos/15fps_videos/"
output_path = "C:/Users/Kanav/Documents/Dissertation/Parkinsons_Disease/Data/parkinsons_videos/cropped_videos/"

bounding_data = pd.read_csv("C:/Users/Kanav/Documents/Dissertation/Parkinsons_Disease/Codes/Boundingbox_dim_files/file_with_updated_centres.csv")

video_counter=0
for file in os.listdir(video_source):
    video_counter+=1
    video_name = os.fsdecode(file)
    print(video_counter, "started. Name is ", video_name)
    if video_name.endswith(".mp4"):
        # Open the video
        cap = cv2.VideoCapture(video_source+ video_name)

        # Initialize frame counter
        cnt = 0

        # Some characteristics from the original video
        w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #w_frame, h_frame = 320,180
        input_fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print("input fps is ", input_fps)

        # Here you can define your croping values
        x = int(bounding_data.loc[bounding_data["filename"]==video_name,"new_xmin"].values[0])
        y = int(bounding_data.loc[bounding_data["filename"]==video_name,"new_ymin"].values[0])
        w,h = 780,910   # change the values here as per the dimensions of the largest bounding box determined previously 

        # output  
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out_fps = 15 # chnage the fps here as per video requirements
        out = cv2.VideoWriter((output_path+video_name), fourcc, out_fps, (w,h),0) # kanav update - added 0 for grey scale

        # Now we start
        while(cap.isOpened()):
            ret, frame = cap.read()

            cnt += 1 # Counting frames

            # Avoid problems when video finish
            if ret==True:
                # We will reduce the FPS rate to 30 & want fixed length video so will write alternate frame. update: 30/06 - this is not needed as we are using 15fps videos, but still keeping it. No harm 
                if input_fps >58 and cnt%2 ==0:
                    # Croping the frame
                    crop_frame = frame[y:y+h, x:x+w] ##kanav update
                    # Percentage
                    #xx = cnt *100/frames
                   # print(int(xx),'%')

                    #grey scale
                    crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)

                    # Here you save all the video.
                    out.write(crop_frame)

                    # Just to see the video in real time          
                    #cv2.imshow('frame',frame)
                    #cv2.imshow('croped',crop_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                # repeating same for any video of 15 fps. Then we dont want to drop frames
                elif input_fps<=20 :
                    crop_frame = frame[y:y+h, x:x+w]
                    crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
                    out.write(crop_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
            else:
                break
    print(video_counter, "is cropped. Name is ", video_name)

cap.release()
out.release()
cv2.destroyAllWindows()

