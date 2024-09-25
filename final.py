import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.pkl', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands = 1)

# {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'}
labels_dict = dict()
ind = 0
for i in range(65, 91):
    labels_dict[ind] = chr(i)
    ind += 1
for i in range(0, 10):
    labels_dict[ind] = str(i)
    ind += 1

print(labels_dict)

predicted_character = ""
phrase = ""

while cv2.waitKey(100)!=27:

    datapoints = []
    x_dp = []
    y_dp = []

    ret, frame = cap.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x_dp.append(hand_landmarks.landmark[i].x)
                y_dp.append(hand_landmarks.landmark[i].y)

            minX = min(x_dp)
            minY = min(y_dp)
            for i in range(len(x_dp)):
                datapoints.append(x_dp[i] - minX)
                datapoints.append(y_dp[i] - minY)
                 

        prediction = model.predict([np.array(datapoints)])

        last_predicted_char = predicted_character
        predicted_character = labels_dict[int(prediction[0])]

        if(last_predicted_char != predicted_character):
            phrase += predicted_character

        frame = cv2.flip(frame, 1)

        cv2.putText(frame, phrase, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        print(phrase)
    else:
        frame = cv2.flip(frame, 1)
        phrase = ""

    cv2.imshow('frame', frame)


cap.release()
cv2.destroyAllWindows()
