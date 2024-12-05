Yoga Pose Detection and Feedback System
1. Introduction
The Yoga Pose Detection and Feedback System is an AI-powered application designed to enhance the yoga experience by providing real-time feedback on pose alignment. Using computer vision techniques, the system detects yoga poses from images and evaluates the alignment of key body joints, offering suggestions for improvement.

2. Approach
The approach taken in this project involves the following steps:

Pose Detection: Utilize a pre-trained model (MediaPipe Pose) to detect key landmarks of the human body in images.
Feedback Mechanism: Calculate angles between specific joints to assess the alignment of yoga poses and provide feedback to the user.
User Interaction: Allow users to upload images of their yoga poses and receive immediate feedback.
3. Data Preprocessing
3.1 Data Collection
For this proof of concept, we used images of yoga poses. Users can upload their own images to the system for analysis. The images should ideally be clear and well-lit to ensure accurate pose detection.

3.2 Image Processing
Color Conversion: The images are converted from BGR (OpenCV default) to RGB format for processing with MediaPipe.
Pose Detection: The MediaPipe Pose model processes the images to identify key landmarks corresponding to various body parts.
4. Model Architecture
4.1 MediaPipe Pose
The MediaPipe Pose model is a lightweight and efficient model designed for real-time pose detection. It uses a combination of:

Convolutional Neural Networks (CNNs): For feature extraction from images.
Landmark Detection: Identifies 33 key points on the human body, including joints and extremities.
4.2 Feedback Calculation
The feedback mechanism involves calculating the angles between specific joints (e.g., shoulder, elbow, wrist) to assess the alignment of the pose. The angle is calculated using the following formula:

[ \text{angle} = \text{atan2}(c_y - b_y, c_x - b_x) - \text{atan2}(a_y - b_y, a_x - b_x) ]

Where:

(a), (b), and (c) are the coordinates of the joints.
5. Results
5.1 Pose Detection
The system successfully detects key landmarks in the uploaded images. The annotated images display the detected landmarks and connections, providing a visual representation of the pose.

5.2 Feedback
The feedback mechanism evaluates the angles between joints and provides suggestions for improvement. For example:

If the angle between the shoulder, elbow, and wrist is less than 160 degrees, the system suggests adjusting the elbow position for better alignment.
If the angle is within the acceptable range, the system confirms that the pose looks good.
5.3 User Experience
Users can easily upload their images and receive immediate feedback, enhancing their yoga practice and helping them improve their poses.

6. Next Steps
To further enhance the Yoga Pose Detection and Feedback System, the following steps are recommended:

Expand Pose Library: Include more yoga poses and corresponding feedback mechanisms to cover a wider range of yoga practices.
Real-Time Feedback: Implement a real-time video analysis feature that allows users to receive feedback while performing poses.
Mobile Application: Develop a mobile application to make the system more accessible to users practicing yoga at home or in studios.
User Profiles: Create user profiles to track progress over time, allowing users to see improvements in their pose alignment.
Integration with Wearables: Explore integration with wearable devices to gather additional data on user performance and provide more personalized feedback.
7. Conclusion
The Yoga Pose Detection and Feedback System demonstrates the potential of AI and computer vision in enhancing the yoga experience. By providing real-time feedback on pose alignment, the system empowers users to improve their practice and achieve better results.

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
# Upload an image
uploaded = files.upload()

# Define pose detection function
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def detect_pose(image):
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Draw the pose annotation on the image
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return image, results.pose_landmarks

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def provide_feedback(landmarks):
    if landmarks:
        # Example: Check the angle between shoulder, elbow, and wrist
        shoulder = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        angle = calculate_angle(shoulder, elbow, wrist)
        if angle < 160:
            print("Adjust your elbow position for better alignment.")
        else:
            print("Your pose looks good!")
    else:
        print("Pose landmarks not detected.")

# Load the uploaded image
image_path = list(uploaded.keys())[0]  # Get the filename of the uploaded image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Error loading image.")
else:
    # Detect pose
    annotated_image, landmarks = detect_pose(image)

    # Display the result
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Provide feedback
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def provide_feedback(landmarks):
    if landmarks:
        # Example: Check the angle between shoulder, elbow, and wrist
        shoulder = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        angle = calculate_angle(shoulder, elbow, wrist)
        if angle < 160:
            print("Adjust your elbow position for better alignment.")
        else:
            print("Your pose looks good!")
    else:
        print("Pose landmarks not detected.")

# Provide feedback
provide_feedback(landmarks)
    
