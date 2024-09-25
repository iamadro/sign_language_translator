import cv2
import os

cam = cv2.VideoCapture(0)
dest = "./new_data"

if not os.path.exists(dest):
    os.makedirs(dest)

#Setting the number of entries to input(num_inputs), and the number of frames to be captured for each entry(dataset_size)
num_inputs = 11
dataset_size = 200

start_from = 0  #from which directory to start adding frames

if os.listdir(dest) != []:
    start_from = int(os.listdir(dest)[-1]) + 1
    print("HEHE")

for i in range(start_from, start_from + num_inputs):
    print("Press 's' to capture")
    if not os.path.exists(os.path.join(dest, str(i))):
        os.makedirs(os.path.join(dest, str(i)))

    while(True):
        success, frame = cam.read()
        cv2.imshow("Inputing Data", frame)
        if cv2.waitKey(25) == ord("s") or cv2.waitKey(25) == ord("S"):
            break
    
    print(f"Capturing for gesture {i}")
    
    for j in range(dataset_size):
        success, frame = cam.read()
        cv2.imshow("Inputing Data", frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(dest, str(i), f"{j}.jpg"), frame)
    print(f"End capturing")

cam.release()
cv2.destroyAllWindows()