import cv2
import os
import time

def capture(hand_type):
    cam = cv2.VideoCapture(0)

    newpath = rf"folder to {hand_type}"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    count = 0
    while(True):
        ret, frame = cam.read()
        cv2.imshow("show", frame)
        filename = os.path.join(newpath, f"{type}_{count}.jpg")
        cv2.imwrite(filename, frame)
        count += 1
        time.sleep(1)
        print(f"Saved {filename}")
        if count == 20:
            break
    cam.release()
    cv2.destroyAllWindows()






