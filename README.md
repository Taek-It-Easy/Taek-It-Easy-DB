# Taek It Easy - MySQL Database
## Pose Estimation Model: MoveNet
### ❔What is MoveNet?
MoveNet is a state-of-the-art human pose estimation model by Google, designed for real-time detection of human joints in images or videos. It efficiently identifies key body points, making it ideal for real-time applications like fitness apps and gesture recognition. Available in versions for speed (Lightning) or accuracy (Thunder), MoveNet is hosted on TensorFlow Hub for easy integration into projects, enabling innovative uses in health monitoring and interactive experiences.

### MoveNet version: singlepose-lightning
A convolutional neural network model that runs on RGB images and predicts human joint locations of a single person. The model is designed to be run in the browser using Tensorflow.js or on devices using TF Lite in real-time, targeting movement/fitness activities. This variant: MoveNet.SinglePose.Lightning is a lower capacity model (compared to MoveNet.SinglePose.Thunder) that can run
(source: https://www.kaggle.com/models/google/movenet/frameworks/tensorFlow2/variations/singlepose-lightning)

### REFERENCE
This project utilizes Google's MoveNet model, which is available on TensorFlow Hub at https://tfhub.dev/google/movenet/singlepose/lightning/4. It was employed for real-time human pose estimation.

## File Descriptions
### MoveNet.py
MoveNet.py provides a function that can use MoveNet to optimize the Taekwondo posture of an image and then return the coordinates. A brief description of each function is given below.

* load_model
    * Load the TensorFlow model for pose estimation.

* process_image
    * Load the image to perform the pose estimation.
    * This function contains a series of processes such as reading image, JPEG Decode, grayscale to RGB conversion.

* run_movenet
    * Extract the coordinates for the keypoint from the processed image.
    * There are a total of 17 key points, including 'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist', 'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'.
    * Key points are calculated as percentages proportional to image size, not pixel coordinates.

### Database.ipynb
The task of inserting into the DB is repeated by changing the direction of the hand and foot, and the file name.

### config.ini
'config.ini' file contains private information for the MySQL Database connection.