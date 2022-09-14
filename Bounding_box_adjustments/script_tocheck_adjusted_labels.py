#!/usr/bin/env python
# coding: utf-8

# In[11]:


# import os
# video_source = "C:/Users/Kanav/Documents/Dissertation/Parkinsons_Disease/Data/parkinsons_videos/labeled_data/"
# file_list = []
# for file_list1 in os.listdir(video_source):
#     print(file_list1)
#     for files in os.listdir(video_source + file_list1 + "/"):
#         #print(files)
#         file_list.append(files)


# In[17]:


# ## Copying excel files 
# import shutil

# video_source = "C:/Users/Kanav/Documents/Dissertation/Parkinsons_Disease/Data/parkinsons_videos/labeled_data/"
# output_location = "C:/Users/Kanav/Documents/Dissertation/Parkinsons_Disease/Data/parkinsons_videos/labeled_data/all_coordinate_files/"
# for file_list1 in os.listdir(video_source):
#     if file_list1.endswith("_labelled"):
#         #print(file_list1)
#         for file_list2 in os.listdir(video_source + file_list1 + "/"):
#             if file_list2.endswith("_lowfps"):
#                 for file_list3 in os.listdir(video_source + file_list1 + "/" + file_list2 + "/" ):
#                     if file_list3.endswith(".csv"):
#                         #print(file_list3)


# In[29]:


# import pandas as pd
# video_source = "C:/Users/Kanav/Documents/Dissertation/Parkinsons_Disease/Data/parkinsons_videos/"
# label_data = pd.read_csv(video_source + "test_file_to_see_updated_labels.csv")

# label_data.iloc[164:,5:7]


# In[ ]:


import numpy as np
import cv2
import tensorflow as tf
import os
from pathlib import Path
import pandas as pd

video_source = "C:/Users/Kanav/Documents/Dissertation/Parkinsons_Disease/Data/parkinsons_videos/test/"
#output_path = "C:/Users/Kanav/Documents/Dissertation/Parkinsons_Disease/Data/parkinsons_videos/cropped_videos/"

label_data = pd.read_csv(video_source + "test_file_to_see_updated_labels.csv")

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
        #print("input fps is ", input_fps)

        print("this is the frame count", round(cap.get(7)))
        #w,h = 780,910   # change the values here as per the dimensions of the largest bounding box determined previously 

        # output  
        #fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        #out_fps = 15 # chnage the fps here as per video requirements
        #out = cv2.VideoWriter((output_path+video_name), fourcc, out_fps, (w,h),0) # kanav update - added 0 for grey scale

        # Now we start
        while (cap.isOpened()):
            ret, frame = cap.read()
            # Here we mark the thumb & finger
            #print(cnt)
        
            # Avoid problems when video finish
            if ret==True :
                x = int(label_data.iloc[cnt,5:9].values[0])
                y = int(label_data.iloc[cnt,5:9].values[1])
                z = int(label_data.iloc[cnt,5:9].values[2])
                w = int(label_data.iloc[cnt,5:9].values[3]) 

                image = cv2.circle(frame, (x,y), radius=0, color=(0, 0, 255), thickness=10)
                image = cv2.circle(frame, (z,w), radius=0, color=(0, 255, 0), thickness=10)
                
                # Just to see the video in real time          
                cv2.imshow('frame',image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                        break                    
            
                cnt += 1 # Counting frames
            
            else:
                break
    print(video_counter, "is cropped. Name is ", video_name)

cap.release()
#out.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[80]:


# from pathlib import Path
# p = Path("C:/Users/Kanav/Documents/Dissertation/Parkinsons_Disease/Codes")
# filename = "kanav" + ".csv"
# dd.to_csv(Path(p,filename), index = False)


# In[5]:


# import numpy as np
# import cv2
# import tensorflow as tf
# import os
# from pathlib import Path
# import pandas as pd

# video_source = "C:/Users/Kanav/Documents/Dissertation/Parkinsons_Disease/Data/parkinsons_videos/60fps_videos/"
# output_path = "C:/Users/Kanav/Documents/Dissertation/Parkinsons_Disease/Data/parkinsons_videos/cropped_videos/"

# bounding_data = pd.read_csv("C:/Users/Kanav/Documents/Dissertation/Parkinsons_Disease/Codes/Boundingbox_dim_files/file_with_updated_centres.csv")

# video_counter=0
# for file in os.listdir(video_source):
#     video_counter+=1
#     video_name = os.fsdecode(file)
#     print(video_counter, "started. Name is ", video_name)
#     if video_name.endswith(".mp4"):
#         # Open the video
#         cap = cv2.VideoCapture(video_source+ video_name)

#         # Initialize frame counter
#         cnt = 0

#         # Some characteristics from the original video
#         w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         #w_frame, h_frame = 320,180
#         fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)

#         # Here you can define your croping values
#         x = int(bounding_data.loc[bounding_data["filename"]==video_name,"new_xmin"].values[0])
#         y = int(bounding_data.loc[bounding_data["filename"]==video_name,"new_ymin"].values[0])
#         w,h = 780,920 
# #        print("this is x",x)
# #        print("this is y",y)


#         # output  
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
#         out_fps = 30
#         out = cv2.VideoWriter((output_path+video_name), fourcc, out_fps, (w,h),0) # kanav update - added 0 for grey scale

#         # Now we start
#         while(cap.isOpened()):
#             ret, frame = cap.read()

#             cnt += 1 # Counting frames

#             # Avoid problems when video finish
#             if ret==True:
#                 # Croping the frame
#                 crop_frame = frame[y:y+h, x:x+w] ##kanav update
#                 # Percentage
#                 xx = cnt *100/frames
#                # print(int(xx),'%')

#                 #grey scale
#                 crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)

#                 # Here you save all the video
#                 out.write(crop_frame)
                
#                 # Just to see the video in real time          
#                 #cv2.imshow('frame',frame)
#                 #cv2.imshow('croped',crop_frame)

#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#             else:
#                 break
#     print(video_counter, "is cropped. Name is ", video_name)

# cap.release()
# out.release()
# cv2.destroyAllWindows()

