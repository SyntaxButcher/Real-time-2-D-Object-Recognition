# Real-time 2-D Object Recognition

This project is a real-time object classification system that uses OpenCV and a simple nearest neighbor algorithm for classification. The system is trained on a set of region features that include Hu moments, the percentage of the region filled, and the height-to-width ratio of the oriented bounding box.

# Key Features

**Real-time Video Capture**: The system captures video in real-time from the default camera device.

**Image Processing**: Each frame of the video is converted to grayscale and thresholded to create a binary image. The binary image is then cleaned using a morphological opening operation.

**Region Identification**: Regions in the cleaned binary image are identified using OpenCV's connected components method.

**Feature Extraction**: For each identified region, several features are computed, including Hu moments, the percentage of the region filled, and the height-to-width ratio of the oriented bounding box.

**Object Classification**: Each region is classified using a simple nearest neighbor algorithm, which finds the most similar instance in the training set and assigns that instance's label to the region.

**Training Mode**: The system has a training mode that allows users to add new instances to the training set. The training set is saved to a CSV file after every update.

# Dependencies
The system depends on OpenCV library (version 4.x).

# Build and Run
The system can be built and run using any C++ compiler that supports C++11. You will need to install OpenCV and ensure that your compiler can find the OpenCV headers and libraries.

# Object detector in action

<img src="https://github.com/SyntaxButcher/Real-time-2-D-Object-Recognition/blob/main/Results/Airpods.png" width="50%" height="50%">
<img src="https://github.com/SyntaxButcher/Real-time-2-D-Object-Recognition/blob/main/Results/Phone.png" width="50%" height="50%">
<img src="https://github.com/SyntaxButcher/Real-time-2-D-Object-Recognition/blob/main/Results/webcamRemote.png" width="50%" height="50%">


 
