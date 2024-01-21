from flask import Flask, request, jsonify
import cv2
import numpy as np
from time import time
import mediapipe as mp
import pyttsx3

app = Flask(_name_)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.2, model_complexity=2)

mp_drawing = mp.solutions.drawing_utils

engine = pyttsx3.init()

last_detected_pose = 'Unknown Pose'

def detectPose(image, pose):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))

    return output_image, landmarks

def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

def classifyPose(landmarks, output_image):
    global last_detected_pose
    label = 'Unknown Pose'
    color = (0, 0, 255)

    # Pose classification logic remains the same

    if label != 'Unknown Pose':
        color = (0, 255, 0)
        if label != last_detected_pose:
            last_detected_pose = label

    return output_image, label

@app.route('/detect_pose', methods=['POST'])
def detect_pose():
    try:
        # Get the image from the request
        image_data = request.files['image'].read()
        nparr = np.fromstring(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process the image and detect pose
        output_image, landmarks = detectPose(image, pose)

        # Classify the pose
        output_image, label = classifyPose(landmarks, output_image)

        # Prepare the response
        _, img_encoded = cv2.imencode('.jpg', output_image)
        img_bytes = img_encoded.tobytes()
        
        response_data = {
            'label': label,
            'image': img_bytes
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if _name_ == '_main_':
    app.run(debug=True)