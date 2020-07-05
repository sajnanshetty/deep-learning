## Task1
OpenCV Yolo: [SOURCE ](https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/)

1. Run this above code on your laptop or Colab. 
2. Take an image of yourself, holding another object which is there in COCO data set (search for COCO classes to learn). 
3. Run this image through the code above. 
4. Upload the link to GitHub implementation of this
5. Upload the annotated image by YOLO. 

### Summary of Annotated Images
The class "**bottle**" detected correctly.

<img src="https://github.com/sajnanshetty/deep-learning/blob/master/s13/yolo_opencv/images/image4_detect.PNG">

The classes '**pen**' and '**hand**' are not detected and class '**tab**' is detected as '**tvmonitor**'

<img src="https://github.com/sajnanshetty/deep-learning/blob/master/s13/yolo_opencv/images/image3_detect.PNG">

## Task B
Training Custom Dataset on Colab for YoloV3
1.  Refer to this File: [LINK ](https://github.com/sajnanshetty/deep-learning/blob/master/s13/yolov3_trained/yolo_v3.ipynb)
2.  Refer to this GitHub [Repo](https://github.com/theschoolofai/YoloV3)
3.  Collect a dataset of 500 images and annotate it. **Please select a class for which you can find a YouTube video as well.** Steps are explained in the readme.md file on GitHub.
4.  Once done:
    1.  [Download](https://www.y2mate.com/en19)  a very small (~10-30sec) video from youtube which shows your class.
    2.  Use [ffmpeg](https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence)  to extract frames from the video.
    3.  Inter on these images using detect.py file.**Modify** detect.py if required.
        `python detect.py --conf-thres 0.3 --output output_folder_name`
    4.  Use [ffmpeg](https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence)  to converted the files to video

###	Summary of training custom dataset result using YoloV3:
- Created a custom dataset of 500 images of **tom & jerry**.
- Annotated the custom dataset using [https://github.com/miki998/YoloV3_Annotation_Tool](https://github.com/miki998/YoloV3_Annotation_Tool) for class **jerry**
- Trained YoloV3 using [https://github.com/sajnanshetty/deep-learning/tree/master/s13/yolov3_trained](https://github.com/sajnanshetty/deep-learning/tree/master/s13/yolov3_trained) which was forked from [https://github.com/theschoolofai/YoloV3](https://github.com/theschoolofai/YoloV3)

### Results

#### Visualise Train Batch
<img src="https://github.com/sajnanshetty/deep-learning/blob/master/s13/yolov3_trained/images/train_batch.PNG">

#### Detecting jerry few samples images from video
<img src="https://github.com/sajnanshetty/deep-learning/blob/master/s13/yolov3_trained/images/detected_images.PNG" height="300">

#### YouTube Video
A video of jerry detector found on YouTube was passed through the trained model. 
The video can be found at [https://youtu.be/HbtuBvCc3AU](https://youtu.be/HbtuBvCc3AU)



