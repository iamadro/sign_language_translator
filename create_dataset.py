import os
import pickle
import datetime as dt
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=1)

DATA_DIR = './new_data'

data = []
labels = []
for letter in os.listdir(DATA_DIR):
    print("Working on : ", letter)
    for image in os.listdir(os.path.join(DATA_DIR, letter)):
        datapoints = []

        xpoints = []
        ypoints = []

        img = cv2.imread(os.path.join(DATA_DIR, letter, image))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    xpoints.append(hand_landmarks.landmark[i].x)
                    ypoints.append(hand_landmarks.landmark[i].y)

                minX = min(xpoints)
                minY = min(ypoints)
                for i in range(len(xpoints)):
                    datapoints.append(xpoints[i] - minX)
                    datapoints.append(ypoints[i] - minY)
                    
            data.append(datapoints)
            labels.append(letter)
    print("Done:", dt.datetime.now())

f = open('data.pkl', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
